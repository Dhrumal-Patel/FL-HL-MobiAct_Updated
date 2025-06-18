import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from model import LSTMModel

def save_model_to_csv(model, file_path, config):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    state_dict = model.state_dict()
    flattened_params = {}
    for name, param in state_dict.items():
        flattened_params[name] = param.cpu().numpy().flatten()
    max_length = max(len(arr) for arr in flattened_params.values())
    for name in flattened_params:
        flattened_params[name] = np.pad(
            flattened_params[name],
            (0, max_length - len(flattened_params[name])),
            mode='constant',
            constant_values=0
        )
    df = pd.DataFrame(flattened_params)
    df.to_csv(file_path, index=False)
    print(f"Model saved to {file_path}")

def load_model_from_csv(model, file_path, device, config):
    if not os.path.exists(file_path):
        print(f"Model file {file_path} does not exist. Initializing with default weights.")
        return
    try:
        df = pd.read_csv(file_path)
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in df.columns:
                param_data = df[name].values[:param.numel()].reshape(param.shape)
                param.copy_(torch.tensor(param_data, dtype=torch.float32).to(device))
            else:
                print(f"Parameter {name} not found in CSV. Keeping default initialization.")
        model.load_state_dict(state_dict)
        print(f"Model loaded from {file_path}")
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")

class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.models = {
            'binary': LSTMModel(
                input_size=9,
                hidden_size=config.HIDDEN_SIZE_BINARY,
                num_layers=config.NUM_LAYERS,
                num_classes=2
            ).to(self.device),
            'fall': LSTMModel(
                input_size=9,
                hidden_size=config.HIDDEN_SIZE_MULTICLASS,
                num_layers=config.NUM_LAYERS,
                num_classes=len(config.FALL_SCENARIOS)
            ).to(self.device),
            'non_fall': LSTMModel(
                input_size=9,
                hidden_size=config.HIDDEN_SIZE_MULTICLASS,
                num_layers=config.NUM_LAYERS,
                num_classes=len(config.NON_FALL_SCENARIOS)
            ).to(self.device)
        }
    
    def aggregate(self, client_ids, algorithm, client_sample_counts):
        if not client_ids:
            print("No clients provided for aggregation")
            return
        
        averaged_models = {
            'binary': {name: torch.zeros_like(param) for name, param in self.models['binary'].state_dict().items()},
            'fall': {name: torch.zeros_like(param) for name, param in self.models['fall'].state_dict().items()},
            'non_fall': {name: torch.zeros_like(param) for name, param in self.models['non_fall'].state_dict().items()}
        }
        
        valid_client_count = 0
        total_binary_samples = 0
        total_fall_samples = 0
        total_non_fall_samples = 0
        
        # Compute total samples for each model
        for client_id in client_ids:
            if client_id in client_sample_counts:
                total_binary_samples += client_sample_counts[client_id]['total']
                total_fall_samples += client_sample_counts[client_id]['fall']
                total_non_fall_samples += client_sample_counts[client_id]['non_fall']
        
        print(f"Total binary samples: {total_binary_samples}, Total fall samples: {total_fall_samples}, Total non-fall samples: {total_non_fall_samples}")
        
        for client_id in client_ids:
            client_path = os.path.join(self.config.MODEL_SAVE_FOLDER, f'client_{client_id}')
            if not os.path.exists(client_path):
                print(f"Client {client_id} model directory does not exist")
                continue
            
            valid_client_count += 1
            
            for model_name in ['binary', 'fall', 'non_fall']:
                client_model_path = os.path.join(client_path, f'{model_name}_params.csv')
                if not os.path.exists(client_model_path):
                    print(f"Client {client_id} {model_name} model file does not exist")
                    continue
                
                # Determine client weight based on model type
                if algorithm == 'weighted_fedavg':
                    if model_name == 'binary':
                        client_weight = (client_sample_counts[client_id]['total'] / total_binary_samples
                                         if total_binary_samples > 0 else 1.0 / len(client_ids))
                    elif model_name == 'fall':
                        client_weight = (client_sample_counts[client_id]['fall'] / total_fall_samples
                                         if total_fall_samples > 0 and client_sample_counts[client_id]['fall'] > 0
                                         else 0.0)
                    elif model_name == 'non_fall':
                        client_weight = (client_sample_counts[client_id]['non_fall'] / total_non_fall_samples
                                         if total_non_fall_samples > 0 and client_sample_counts[client_id]['non_fall'] > 0
                                         else 0.0)
                else:  # fedavg
                    client_weight = 1.0 / len(client_ids)
                
                if client_weight == 0.0:
                    print(f"Skipping client {client_id} for {model_name} (no relevant samples)")
                    continue
                
                print(f"Client {client_id} {model_name} weight: {client_weight:.4f}")
                
                temp_model = LSTMModel(
                    input_size=9,
                    hidden_size=self.config.HIDDEN_SIZE_BINARY if model_name == 'binary' else self.config.HIDDEN_SIZE_MULTICLASS,
                    num_layers=self.config.NUM_LAYERS,
                    num_classes=2 if model_name == 'binary' else
                               len(self.config.FALL_SCENARIOS) if model_name == 'fall' else
                               len(self.config.NON_FALL_SCENARIOS)
                ).to(self.device)
                
                load_model_from_csv(temp_model, client_model_path, self.device, self.config)
                
                for name, param in temp_model.state_dict().items():
                    averaged_models[model_name][name] += client_weight * param
        
        if valid_client_count == 0:
            print("No valid clients for aggregation")
            return
        
        for model_name in averaged_models:
            for name in averaged_models[model_name]:
                self.models[model_name].state_dict()[name].copy_(averaged_models[model_name][name])
            
            save_model_to_csv(
                self.models[model_name],
                os.path.join(self.config.MODEL_SAVE_FOLDER, f'global_{model_name}_params.csv'),
                self.config
            )
