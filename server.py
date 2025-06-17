import os
import torch
import csv
import numpy as np
from model import LSTMModel

def save_model_to_csv(model, file_path, config):
    """Save model parameters and metadata to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    state_dict = model.state_dict()
    
    # Prepare metadata
    metadata = {
        'input_size': 9,  # Fixed based on dataset (acc_x, acc_y, etc.)
        'hidden_size': config.HIDDEN_SIZE_BINARY if 'binary' in file_path else config.HIDDEN_SIZE_MULTICLASS,
        'num_layers': config.NUM_LAYERS,
        'num_classes': (2 if 'binary' in file_path else
                        len(config.FALL_SCENARIOS) if 'fall' in file_path else
                        len(config.NON_FALL_SCENARIOS))
    }
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write metadata
        writer.writerow(['__metadata__'] + [f"{k}={v}" for k, v in metadata.items()])
        # Write parameters
        for name, param in state_dict.items():
            param_flat = param.cpu().numpy().flatten()
            writer.writerow([name] + param_flat.tolist())
    # print(f"[save_model_to_csv] Saved model to {file_path} with metadata: {metadata}")

def load_model_from_csv(model, file_path, device, config):
    """Load model parameters from a CSV file, checking metadata compatibility."""
    if not os.path.exists(file_path):
        print(f"[load_model_from_csv] File {file_path} not found, skipping load")
        return
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            metadata = {}
            loaded_params = {}
            
            # Read CSV rows
            for row in reader:
                if row[0] == '__metadata__':
                    # Parse metadata
                    for item in row[1:]:
                        key, value = item.split('=')
                        metadata[key] = int(value)
                else:
                    # Parse parameters
                    param_name = row[0]
                    param_data = np.array([float(x) for x in row[1:]], dtype=np.float32)
                    loaded_params[param_name] = torch.tensor(param_data)
            
            # Expected metadata based on model type
            expected_metadata = {
                'input_size': 9,
                'hidden_size': config.HIDDEN_SIZE_BINARY if 'binary' in file_path else config.HIDDEN_SIZE_MULTICLASS,
                'num_layers': config.NUM_LAYERS,
                'num_classes': (2 if 'binary' in file_path else
                                len(config.FALL_SCENARIOS) if 'fall' in file_path else
                                len(config.NON_FALL_SCENARIOS))
            }
            
            # Check metadata compatibility
            if metadata != expected_metadata:
                print(f"[load_model_from_csv] Metadata mismatch in {file_path}. Expected: {expected_metadata}, Got: {metadata}. Skipping load.")
                return
            
            # Load parameters into model
            current_state = model.state_dict()
            for param_name in current_state:
                if param_name in loaded_params:
                    param = loaded_params[param_name]
                    if param.numel() != current_state[param_name].numel():
                        raise ValueError(f"[load_model_from_csv] Shape mismatch for {param_name}: expected {current_state[param_name].shape}, got {param.shape}")
                    current_state[param_name].copy_(param.view(current_state[param_name].shape).to(device))
            model.load_state_dict(current_state)
            # print(f"[load_model_from_csv] Loaded model from {file_path} with metadata: {metadata}")
    
    except Exception as e:
        print(f"[load_model_from_csv] Error loading model from {file_path}: {str(e)}")
        raise

class FederatedServer:
    def __init__(self, config):
        """Initialize the server with global models."""
        self.config = config
        self.models = {
            'binary': LSTMModel(
                input_size=9,
                hidden_size=config.HIDDEN_SIZE_BINARY,
                num_layers=config.NUM_LAYERS,
                num_classes=2
            ).to(config.DEVICE),
            'fall': LSTMModel(
                input_size=9,
                hidden_size=config.HIDDEN_SIZE_MULTICLASS,
                num_layers=config.NUM_LAYERS,
                num_classes=len(config.FALL_SCENARIOS)
            ).to(config.DEVICE),
            'non_fall': LSTMModel(
                input_size=9,
                hidden_size=config.HIDDEN_SIZE_MULTICLASS,
                num_layers=config.NUM_LAYERS,
                num_classes=len(config.NON_FALL_SCENARIOS)
            ).to(config.DEVICE)
        }
        # Debug: Print model parameter shapes
        # for model_name, model in self.models.items():
            # print(f"[FederatedServer] Initialized {model_name} model with lstm.weight_ih_l0 shape: {model.state_dict()['lstm.weight_ih_l0'].shape}")

    def aggregate(self, client_ids, algorithm='fedavg', client_sample_counts=None):
        """
        Aggregate client models using FedAvg or Weighted FedAvg.
        Args:
            client_ids: List of client IDs to aggregate.
            algorithm: Aggregation method ('fedavg' or 'weighted_fedavg').
            client_sample_counts: Dict mapping client IDs to number of samples (optional for weighted_fedavg).
        """
        try:
            # print(f"[FederatedServer] Aggregating models for {len(client_ids)} clients: {client_ids} using {algorithm}")
            
            # Initialize dictionaries to store aggregated parameters for each model
            averaged_models = {
                'binary': {name: torch.zeros_like(param) for name, param in self.models['binary'].state_dict().items()},
                'fall': {name: torch.zeros_like(param) for name, param in self.models['fall'].state_dict().items()},
                'non_fall': {name: torch.zeros_like(param) for name, param in self.models['non_fall'].state_dict().items()}
            }
            
            # Calculate total samples for weighted averaging
            total_samples = sum(client_sample_counts.values()) if client_sample_counts and algorithm == 'weighted_fedavg' else len(client_ids)
            
            # Track number of valid clients for normalization
            valid_client_count = 0
            
            for client_id in client_ids:
                client_path = os.path.join(self.config.MODEL_SAVE_FOLDER, f'client_{client_id}')
                if not os.path.exists(client_path):
                    print(f"[FederatedServer] Warning: Client directory {client_path} does not exist. Skipping client {client_id}")
                    continue
                
                # Determine the weight for this client
                client_weight = (client_sample_counts[client_id] / total_samples if client_sample_counts and algorithm == 'weighted_fedavg'
                                else 1.0 / len(client_ids))
                # print(f"[FederatedServer] Client {client_id} weight: {client_weight:.4f} (samples: {client_sample_counts.get(client_id, 'N/A')})")
                
                for model_name in averaged_models:
                    client_model_path = os.path.join(client_path, f'{model_name}_params.csv')
                    if not os.path.exists(client_model_path):
                        print(f"[FederatedServer] Warning: Model file {client_model_path} does not exist. Skipping model {model_name} for client {client_id}")
                        continue
                    
                    # Load client parameters into a temporary model
                    temp_model = LSTMModel(
                        input_size=9,
                        hidden_size=self.config.HIDDEN_SIZE_BINARY if model_name == 'binary' else self.config.HIDDEN_SIZE_MULTICLASS,
                        num_layers=self.config.NUM_LAYERS,
                        num_classes=2 if model_name == 'binary' else
                                   len(self.config.FALL_SCENARIOS) if model_name == 'fall' else
                                   len(self.config.NON_FALL_SCENARIOS)
                    ).to(self.config.DEVICE)
                    load_model_from_csv(temp_model, client_model_path, self.config.DEVICE, self.config)
                    
                    # Add weighted parameters to the aggregated model
                    for name, param in temp_model.state_dict().items():
                        averaged_models[model_name][name] += client_weight * param
                    # print(f"[FederatedServer] Aggregated {model_name} model for client {client_id}")
                
                valid_client_count += 1
            
            if valid_client_count == 0:
                print(f"[FederatedServer] Error: No valid client models to aggregate")
                return
            
            # Update global model with aggregated parameters
            for model_name in averaged_models:
                # Compute norm of aggregated parameters for debugging
                norm = sum(torch.norm(param.view(-1), p=2).item() ** 2 for param in averaged_models[model_name].values()) ** 0.5
                # print(f"[FederatedServer] {model_name} aggregated model norm: {norm:.4f}")
                
                # Update global model
                for name in averaged_models[model_name]:
                    self.models[model_name].state_dict()[name].copy_(averaged_models[model_name][name])
                
                # Save global model
                save_model_to_csv(
                    self.models[model_name],
                    os.path.join(self.config.MODEL_SAVE_FOLDER, f'global_{model_name}_params.csv'),
                    self.config
                )
                # print(f"[FederatedServer] Saved aggregated {model_name} model to global_{model_name}_params.csv")
        
        except Exception as e:
            print(f"[FederatedServer] Error during aggregation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise