import os
import json
import csv
import numpy as np
from config import Config
from dataset import prepare_federated_data, verify_dataset
from client import FederatedClient
from server import FederatedServer, load_model_from_csv
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from model import LSTMModel

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_keys_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_keys_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_json_serializable(item) for item in obj]
    return obj

def compute_model_norm(model):
    """Compute the L2 norm of model parameters for debugging."""
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param, p=2).item() ** 2
    return np.sqrt(norm)

def evaluate_global_model(server, loader, config, model_name='test'):
    global_model = {
        'binary': LSTMModel(9, config.HIDDEN_SIZE_BINARY, config.NUM_LAYERS, 2).to(config.DEVICE),
        'fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.FALL_SCENARIOS)).to(config.DEVICE),
        'non_fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.NON_FALL_SCENARIOS)).to(config.DEVICE)
    }
    
    for model_name_key in global_model:
        load_model_from_csv(
            global_model[model_name_key],
            os.path.join(config.MODEL_SAVE_FOLDER, f'global_{model_name_key}_params.csv'),
            config.DEVICE,
            config
        )
        # Debug: Print model norm
        norm = compute_model_norm(global_model[model_name_key])
        print(f"[Debug] {model_name_key} model norm after loading: {norm:.4f}")
    
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'three_models': {
            'binary_correct': 0,
            'fall_correct': 0,
            'non_fall_correct': 0,
            'total': 0,
            'fall_total': 0,
            'non_fall_total': 0,
            'binary_preds': [],
            'binary_targets': [],
            'fall_preds': [],
            'fall_targets': [],
            'non_fall_preds': [],
            'non_fall_targets': [],
            'binary_class_counts': {},
            'fall_class_counts': {},
            'non_fall_class_counts': {}
        },
        'two_models': {
            'binary_acc': 0.0,
            'binary_weighted_acc': 0.0,
            'binary_precision': 0.0,
            'binary_recall': 0.0,
            'binary_f1': 0.0,
            'multiclass_acc': 0.0,
            'multiclass_weighted_acc': 0.0,
            'multiclass_precision': 0.0,
            'multiclass_recall': 0.0,
            'multiclass_f1': 0.0
        }
    }
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            binary_targets, multi_targets = targets[:, 0], targets[:, 1]
            
            binary_out = global_model['binary'](inputs).to(config.DEVICE)
            _, binary_preds = torch.max(binary_out, 1)
            metrics['three_models']['binary_correct'] += (binary_preds == binary_targets).sum().item()
            metrics['three_models']['binary_preds'].extend(binary_preds.tolist())
            metrics['three_models']['binary_targets'].extend(binary_targets.tolist())
            
            for target in binary_targets.cpu().numpy():
                metrics['three_models']['binary_class_counts'][int(target)] = metrics['three_models']['binary_class_counts'].get(int(target), 0) + 1
            
            fall_mask = binary_targets == 0
            if fall_mask.any():
                fall_out = global_model['fall'](inputs[fall_mask]).to(config.DEVICE)
                _, fall_preds = torch.max(fall_out, 1)
                metrics['three_models']['fall_correct'] += (fall_preds == multi_targets[fall_mask]).sum().item()
                metrics['three_models']['fall_total'] += fall_mask.sum().item()
                metrics['three_models']['fall_preds'].extend(fall_preds.tolist())
                metrics['three_models']['fall_targets'].extend(multi_targets[fall_mask].tolist())
                for target in multi_targets[fall_mask].cpu().numpy():
                    metrics['three_models']['fall_class_counts'][int(target)] = metrics['three_models']['fall_class_counts'].get(int(target), 0) + 1
            
            non_fall_mask = binary_targets == 1
            if non_fall_mask.any():
                non_fall_out = global_model['non_fall'](inputs[non_fall_mask]).to(config.DEVICE)
                _, non_fall_preds = torch.max(non_fall_out, 1)
                metrics['three_models']['non_fall_correct'] += (non_fall_preds == multi_targets[non_fall_mask]).sum().item()
                metrics['three_models']['non_fall_total'] += non_fall_mask.sum().item()
                metrics['three_models']['non_fall_preds'].extend(non_fall_preds.tolist())
                metrics['three_models']['non_fall_targets'].extend(multi_targets[non_fall_mask].tolist())
                for target in multi_targets[non_fall_mask].cpu().numpy():
                    metrics['three_models']['non_fall_class_counts'][int(target)] = metrics['three_models']['non_fall_class_counts'].get(int(target), 0) + 1
            
            metrics['three_models']['total'] += len(targets)
    
    # Compute three-model metrics
    if metrics['three_models']['binary_class_counts']:
        binary_weights = np.array([metrics['three_models']['binary_class_counts'].get(i, 0) for i in range(2)])
        binary_weights = binary_weights / binary_weights.sum() if binary_weights.sum() > 0 else np.ones(2) / 2
        binary_correct_weighted = 0
        for i in range(2):
            correct_count = sum(1 for pred, target in zip(metrics['three_models']['binary_preds'], metrics['three_models']['binary_targets']) if pred == target == i)
            binary_correct_weighted += correct_count * binary_weights[i]
        metrics['three_models']['binary_weighted_acc'] = binary_correct_weighted / max(1, metrics['three_models']['total'])
    else:
        metrics['three_models']['binary_weighted_acc'] = 0.0
    
    if metrics['three_models']['fall_class_counts']:
        fall_weights = np.array([metrics['three_models']['fall_class_counts'].get(i, 0) for i in range(len(config.FALL_SCENARIOS))])
        fall_weights = fall_weights / fall_weights.sum() if fall_weights.sum() > 0 else np.ones(len(config.FALL_SCENARIOS)) / len(config.FALL_SCENARIOS)
        fall_correct_weighted = 0
        for i in range(len(config.FALL_SCENARIOS)):
            correct_count = sum(1 for pred, target in zip(metrics['three_models']['fall_preds'], metrics['three_models']['fall_targets']) if pred == target == i)
            fall_correct_weighted += correct_count * fall_weights[i]
        metrics['three_models']['fall_weighted_acc'] = fall_correct_weighted / max(1, metrics['three_models']['fall_total'])
    else:
        metrics['three_models']['fall_weighted_acc'] = 0.0
    
    if metrics['three_models']['non_fall_class_counts']:
        non_fall_weights = np.array([metrics['three_models']['non_fall_class_counts'].get(i, 0) for i in range(len(config.NON_FALL_SCENARIOS))])
        non_fall_weights = non_fall_weights / non_fall_weights.sum() if non_fall_weights.sum() > 0 else np.ones(len(config.NON_FALL_SCENARIOS)) / len(config.NON_FALL_SCENARIOS)
        non_fall_correct_weighted = 0
        for i in range(len(config.NON_FALL_SCENARIOS)):
            correct_count = sum(1 for pred, target in zip(metrics['three_models']['non_fall_preds'], metrics['three_models']['non_fall_targets']) if pred == target == i)
            non_fall_correct_weighted += correct_count * non_fall_weights[i]
        metrics['three_models']['non_fall_weighted_acc'] = non_fall_correct_weighted / max(1, metrics['three_models']['non_fall_total'])
    else:
        metrics['three_models']['non_fall_weighted_acc'] = 0.0
    
    metrics['three_models']['binary_acc'] = metrics['three_models']['binary_correct'] / max(1, metrics['three_models']['total'])
    metrics['three_models']['fall_acc'] = metrics['three_models']['fall_correct'] / max(1, metrics['three_models']['fall_total'])
    metrics['three_models']['non_fall_acc'] = metrics['three_models']['non_fall_correct'] / max(1, metrics['three_models']['non_fall_total'])
    
    if metrics['three_models']['binary_targets']:
        binary_prf = precision_recall_fscore_support(
            metrics['three_models']['binary_targets'], metrics['three_models']['binary_preds'], average='weighted', zero_division=0)
        metrics['three_models']['binary_precision'] = binary_prf[0]
        metrics['three_models']['binary_recall'] = binary_prf[1]
        metrics['three_models']['binary_f1'] = binary_prf[2]
    else:
        metrics['three_models']['binary_precision'] = 0.0
        metrics['three_models']['binary_recall'] = 0.0
        metrics['three_models']['binary_f1'] = 0.0
    
    if metrics['three_models']['fall_targets']:
        fall_prf = precision_recall_fscore_support(
            metrics['three_models']['fall_targets'], metrics['three_models']['fall_preds'], average='weighted', zero_division=0)
        metrics['three_models']['fall_precision'] = fall_prf[0]
        metrics['three_models']['fall_recall'] = fall_prf[1]
        metrics['three_models']['fall_f1'] = fall_prf[2]
    else:
        metrics['three_models']['fall_precision'] = 0.0
        metrics['three_models']['fall_recall'] = 0.0
        metrics['three_models']['fall_f1'] = 0.0
    
    if metrics['three_models']['non_fall_targets']:
        non_fall_prf = precision_recall_fscore_support(
            metrics['three_models']['non_fall_targets'], metrics['three_models']['non_fall_preds'], average='weighted', zero_division=0)
        metrics['three_models']['non_fall_precision'] = non_fall_prf[0]
        metrics['three_models']['non_fall_recall'] = non_fall_prf[1]
        metrics['three_models']['non_fall_f1'] = non_fall_prf[2]
    else:
        metrics['three_models']['non_fall_precision'] = 0.0
        metrics['three_models']['non_fall_recall'] = 0.0
        metrics['three_models']['non_fall_f1'] = 0.0
    
    # Compute two-model metrics
    metrics['two_models']['binary_acc'] = metrics['three_models']['binary_acc']
    metrics['two_models']['binary_weighted_acc'] = metrics['three_models']['binary_weighted_acc']
    metrics['two_models']['binary_precision'] = metrics['three_models']['binary_precision']
    metrics['two_models']['binary_recall'] = metrics['three_models']['binary_recall']
    metrics['two_models']['binary_f1'] = metrics['three_models']['binary_f1']
    metrics['two_models']['multiclass_acc'] = (metrics['three_models']['fall_acc'] + metrics['three_models']['non_fall_acc']) / 2
    metrics['two_models']['multiclass_weighted_acc'] = (metrics['three_models']['fall_weighted_acc'] + metrics['three_models']['non_fall_weighted_acc']) / 2
    metrics['two_models']['multiclass_precision'] = (metrics['three_models']['fall_precision'] + metrics['three_models']['non_fall_precision']) / 2
    metrics['two_models']['multiclass_recall'] = (metrics['three_models']['fall_recall'] + metrics['three_models']['non_fall_recall']) / 2
    metrics['two_models']['multiclass_f1'] = (metrics['three_models']['fall_f1'] + metrics['three_models']['non_fall_f1']) / 2
    
    return convert_keys_to_json_serializable(metrics)

if __name__ == "__main__":
    overlap_values = [0.0]
    num_clients_values = [3]
    verify_dataset(Config(overlap=0.0))
    
    for num_clients in num_clients_values:
        for overlap in overlap_values:
            print(f"\n=== Running Experiments with Overlap = {overlap}, Num Clients = {num_clients} ===")
            config = Config(overlap=overlap, num_clients=num_clients)
            os.makedirs(config.SAVE_FOLDER, exist_ok=True)
            os.makedirs(config.MODEL_SAVE_FOLDER, exist_ok=True)
            
            # Prepare data
            try:
                client_datasets, val_dataset, test_dataset = prepare_federated_data(config)
                print(f"Prepared {len(client_datasets)} client datasets, validation dataset, and test dataset.")
            except Exception as e:
                print(f"Error preparing datasets: {e}")
                continue
            
            for algorithm in ['fedavg', 'weighted_fedavg']:  # FedProx is commented out in original
                print(f"\n=== Running {algorithm.upper()} with Overlap = {overlap}, Num Clients = {num_clients} ===")
                server = FederatedServer(config)
                val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
                algorithm_folder = os.path.join(config.SAVE_FOLDER, algorithm)
                os.makedirs(algorithm_folder, exist_ok=True)
                
                # Initialize global model for norm checking
                global_model = {
                    'binary': LSTMModel(9, config.HIDDEN_SIZE_BINARY, config.NUM_LAYERS, 2).to(config.DEVICE),
                    'fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.FALL_SCENARIOS)).to(config.DEVICE),
                    'non_fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.NON_FALL_SCENARIOS)).to(config.DEVICE)
                }
                
                for round_num in range(config.COMMUNICATION_ROUNDS):
                    print(f"\n=== {algorithm.upper()} Round {round_num + 1}/{config.COMMUNICATION_ROUNDS} ===")
                    # Debug: Check global model norm before round
                    for model_name_key in global_model:
                        load_model_from_csv(
                            global_model[model_name_key],
                            os.path.join(config.MODEL_SAVE_FOLDER, f'global_{model_name_key}_params.csv'),
                            config.DEVICE,
                            config
                        )
                        norm_before = compute_model_norm(global_model[model_name_key])
                        print(f"[Debug] Round {round_num + 1}, {model_name_key} norm before training: {norm_before:.4f}")
                    
                    client_batches = [client_datasets[i:i+44] for i in range(0, len(client_datasets), 44)]
                    
                    for batch_idx, batch_datasets in enumerate(client_batches):
                        print(f"\nProcessing client batch {batch_idx + 1}/{len(client_batches)}")
                        client_metrics = {}
                        client_sample_counts = {}
                        results = []
                        
                        # Train each client
                        for client_id, client_data in enumerate(batch_datasets):
                            print(f"Training client {client_id}")
                            try:
                                client = FederatedClient(client_id, client_data, client_data, config)
                                # Explicitly load global model
                                client.load_global_model(config.MODEL_SAVE_FOLDER)
                                metrics, num_samples = client.train(config.MODEL_SAVE_FOLDER, algorithm, config.CLIENT_EPOCHS)
                                client_metrics[client_id] = convert_keys_to_json_serializable(metrics)
                                client_sample_counts[client_id] = num_samples
                                results.append({
                                    'client_id': client_id,
                                    'num_samples': num_samples,
                                    'metrics': metrics
                                })
                            except Exception as e:
                                print(f"Error training client {client_id}: {e}")
                                continue
                        
                        # Aggregate client models
                        if client_metrics:
                            try:
                                server.aggregate(list(client_metrics.keys()), algorithm, client_sample_counts)
                            except Exception as e:
                                print(f"Error during aggregation: {e}")
                                continue
                        else:
                            print("No valid client results for aggregation")
                            continue
                        
                        # Debug: Check global model norm after aggregation
                        for model_name_key in global_model:
                            load_model_from_csv(
                                global_model[model_name_key],
                                os.path.join(config.MODEL_SAVE_FOLDER, f'global_{model_name_key}_params.csv'),
                                config.DEVICE,
                                config
                            )
                            norm_after = compute_model_norm(global_model[model_name_key])
                            print(f"[Debug] Round {round_num + 1}, {model_name_key} norm after aggregation: {norm_after:.4f}")
                        
                        # Compute average client metrics
                        avg_metrics = {
                            'three_models': {
                                'binary_acc': 0.0,
                                'binary_weighted_acc': 0.0,
                                'binary_precision': 0.0,
                                'binary_recall': 0.0,
                                'binary_f1': 0.0,
                                'fall_acc': 0.0,
                                'fall_weighted_acc': 0.0,
                                'fall_precision': 0.0,
                                'fall_recall': 0.0,
                                'fall_f1': 0.0,
                                'non_fall_acc': 0.0,
                                'non_fall_weighted_acc': 0.0,
                                'non_fall_precision': 0.0,
                                'non_fall_recall': 0.0,
                                'non_fall_f1': 0.0
                            },
                            'two_models': {
                                'binary_acc': 0.0,
                                'binary_weighted_acc': 0.0,
                                'binary_precision': 0.0,
                                'binary_recall': 0.0,
                                'binary_f1': 0.0,
                                'multiclass_acc': 0.0,
                                'multiclass_weighted_acc': 0.0,
                                'multiclass_precision': 0.0,
                                'multiclass_recall': 0.0,
                                'multiclass_f1': 0.0
                            }
                        }
                        num_clients_in_batch = len(client_metrics)
                        if num_clients_in_batch > 0:
                            for metrics in client_metrics.values():
                                avg_metrics['three_models']['binary_acc'] += metrics['binary_acc']
                                avg_metrics['three_models']['binary_weighted_acc'] += metrics['binary_weighted_acc']
                                avg_metrics['three_models']['binary_precision'] += metrics['binary_precision']
                                avg_metrics['three_models']['binary_recall'] += metrics['binary_recall']
                                avg_metrics['three_models']['binary_f1'] += metrics['binary_f1']
                                avg_metrics['three_models']['fall_acc'] += metrics['fall_acc']
                                avg_metrics['three_models']['fall_weighted_acc'] += metrics['fall_weighted_acc']
                                avg_metrics['three_models']['fall_precision'] += metrics['fall_precision']
                                avg_metrics['three_models']['fall_recall'] += metrics['fall_recall']
                                avg_metrics['three_models']['fall_f1'] += metrics['fall_f1']
                                avg_metrics['three_models']['non_fall_acc'] += metrics['non_fall_acc']
                                avg_metrics['three_models']['non_fall_weighted_acc'] += metrics['non_fall_weighted_acc']
                                avg_metrics['three_models']['non_fall_precision'] += metrics['non_fall_precision']
                                avg_metrics['three_models']['non_fall_recall'] += metrics['non_fall_recall']
                                avg_metrics['three_models']['non_fall_f1'] += metrics['non_fall_f1']
                                avg_metrics['two_models']['binary_acc'] += metrics['binary_acc']
                                avg_metrics['two_models']['binary_weighted_acc'] += metrics['binary_weighted_acc']
                                avg_metrics['two_models']['binary_precision'] += metrics['binary_precision']
                                avg_metrics['two_models']['binary_recall'] += metrics['binary_recall']
                                avg_metrics['two_models']['binary_f1'] += metrics['binary_f1']
                                avg_metrics['two_models']['multiclass_acc'] += (metrics['fall_acc'] + metrics['non_fall_acc']) / 2
                                avg_metrics['two_models']['multiclass_weighted_acc'] += (metrics['fall_weighted_acc'] + metrics['non_fall_weighted_acc']) / 2
                                avg_metrics['two_models']['multiclass_precision'] += (metrics['fall_precision'] + metrics['non_fall_precision']) / 2
                                avg_metrics['two_models']['multiclass_recall'] += (metrics['fall_recall'] + metrics['non_fall_recall']) / 2
                                avg_metrics['two_models']['multiclass_f1'] += (metrics['fall_f1'] + metrics['non_fall_f1']) / 2
                            
                            for model_type in avg_metrics:
                                for key in avg_metrics[model_type]:
                                    avg_metrics[model_type][key] /= num_clients_in_batch
                        
                        # Print average client metrics
                        print(f"\nAverage Client Metrics (Three Models) - "
                              f"Binary: Acc={avg_metrics['three_models']['binary_acc']:.4f}, "
                              f"W-Acc={avg_metrics['three_models']['binary_weighted_acc']:.4f}, "
                              f"Prec={avg_metrics['three_models']['binary_precision']:.4f}, "
                              f"Rec={avg_metrics['three_models']['binary_recall']:.4f}, "
                              f"F1={avg_metrics['three_models']['binary_f1']:.4f}, "
                              f"Fall: Acc={avg_metrics['three_models']['fall_acc']:.4f}, "
                              f"W-Acc={avg_metrics['three_models']['fall_weighted_acc']:.4f}, "
                              f"Prec={avg_metrics['three_models']['fall_precision']:.4f}, "
                              f"Rec={avg_metrics['three_models']['fall_recall']:.4f}, "
                              f"F1={avg_metrics['three_models']['fall_f1']:.4f}, "
                              f"Non-Fall: Acc={avg_metrics['three_models']['non_fall_acc']:.4f}, "
                              f"W-Acc={avg_metrics['three_models']['non_fall_weighted_acc']:.4f}, "
                              f"Prec={avg_metrics['three_models']['non_fall_precision']:.4f}, "
                              f"Rec={avg_metrics['three_models']['non_fall_recall']:.4f}, "
                              f"F1={avg_metrics['three_models']['non_fall_f1']:.4f}")
                        print(f"Average Client Metrics (Two Models) - "
                              f"Binary: Acc={avg_metrics['two_models']['binary_acc']:.4f}, "
                              f"W-Acc={avg_metrics['two_models']['binary_weighted_acc']:.4f}, "
                              f"Prec={avg_metrics['two_models']['binary_precision']:.4f}, "
                              f"Rec={avg_metrics['two_models']['binary_recall']:.4f}, "
                              f"F1={avg_metrics['two_models']['binary_f1']:.4f}, "
                              f"Multiclass: Acc={avg_metrics['two_models']['multiclass_acc']:.4f}, "
                              f"W-Acc={avg_metrics['two_models']['multiclass_weighted_acc']:.4f}, "
                              f"Prec={avg_metrics['two_models']['multiclass_precision']:.4f}, "
                              f"Rec={avg_metrics['two_models']['multiclass_recall']:.4f}, "
                              f"F1={avg_metrics['two_models']['multiclass_f1']:.4f}")
                        
                        # Evaluate global model
                        val_metrics = evaluate_global_model(server, val_loader, config, model_name='validation')
                        print(f"\nGlobal Validation Metrics (Three Models) - "
                              f"Binary: Acc={val_metrics['three_models']['binary_acc']:.4f}, "
                              f"W-Acc={val_metrics['three_models']['binary_weighted_acc']:.4f}, "
                              f"Prec={val_metrics['three_models']['binary_precision']:.4f}, "
                              f"Rec={val_metrics['three_models']['binary_recall']:.4f}, "
                              f"F1={val_metrics['three_models']['binary_f1']:.4f}, "
                              f"Fall: Acc={val_metrics['three_models']['fall_acc']:.4f}, "
                              f"W-Acc={val_metrics['three_models']['fall_weighted_acc']:.4f}, "
                              f"Prec={val_metrics['three_models']['fall_precision']:.4f}, "
                              f"Rec={val_metrics['three_models']['fall_recall']:.4f}, "
                              f"F1={val_metrics['three_models']['fall_f1']:.4f}, "
                              f"Non-Fall: Acc={val_metrics['three_models']['non_fall_acc']:.4f}, "
                              f"W-Acc={val_metrics['three_models']['non_fall_weighted_acc']:.4f}, "
                              f"Prec={val_metrics['three_models']['non_fall_precision']:.4f}, "
                              f"Rec={val_metrics['three_models']['non_fall_recall']:.4f}, "
                              f"F1={val_metrics['three_models']['non_fall_f1']:.4f}")
                        print(f"Global Validation Metrics (Two Models) - "
                              f"Binary: Acc={val_metrics['two_models']['binary_acc']:.4f}, "
                              f"W-Acc={val_metrics['two_models']['binary_weighted_acc']:.4f}, "
                              f"Prec={val_metrics['two_models']['binary_precision']:.4f}, "
                              f"Rec={val_metrics['two_models']['binary_recall']:.4f}, "
                              f"F1={val_metrics['two_models']['binary_f1']:.4f}, "
                              f"Multiclass: Acc={val_metrics['two_models']['multiclass_acc']:.4f}, "
                              f"W-Acc={val_metrics['two_models']['multiclass_weighted_acc']:.4f}, "
                              f"Prec={val_metrics['two_models']['multiclass_precision']:.4f}, "
                              f"Rec={val_metrics['two_models']['multiclass_recall']:.4f}, "
                              f"F1={val_metrics['two_models']['multiclass_f1']:.4f}")
                        
                        test_metrics = evaluate_global_model(server, test_loader, config, model_name='test')
                        print(f"\nGlobal Test Metrics (Three Models) - "
                              f"Binary: Acc={test_metrics['three_models']['binary_acc']:.4f}, "
                              f"W-Acc={test_metrics['three_models']['binary_weighted_acc']:.4f}, "
                              f"Prec={test_metrics['three_models']['binary_precision']:.4f}, "
                              f"Rec={test_metrics['three_models']['binary_recall']:.4f}, "
                              f"F1={test_metrics['three_models']['binary_f1']:.4f}, "
                              f"Fall: Acc={test_metrics['three_models']['fall_acc']:.4f}, "
                              f"W-Acc={test_metrics['three_models']['fall_weighted_acc']:.4f}, "
                              f"Prec={test_metrics['three_models']['fall_precision']:.4f}, "
                              f"Rec={test_metrics['three_models']['fall_recall']:.4f}, "
                              f"F1={test_metrics['three_models']['fall_f1']:.4f}, "
                              f"Non-Fall: Acc={test_metrics['three_models']['non_fall_acc']:.4f}, "
                              f"W-Acc={test_metrics['three_models']['non_fall_weighted_acc']:.4f}, "
                              f"Prec={test_metrics['three_models']['non_fall_precision']:.4f}, "
                              f"Rec={test_metrics['three_models']['non_fall_recall']:.4f}, "
                              f"F1={test_metrics['three_models']['non_fall_f1']:.4f}")
                        print(f"Global Test Metrics (Two Models) - "
                              f"Binary: Acc={test_metrics['two_models']['binary_acc']:.4f}, "
                              f"W-Acc={test_metrics['two_models']['binary_weighted_acc']:.4f}, "
                              f"Prec={test_metrics['two_models']['binary_precision']:.4f}, "
                              f"Rec={test_metrics['two_models']['binary_recall']:.4f}, "
                              f"F1={test_metrics['two_models']['binary_f1']:.4f}, "
                              f"Multiclass: Acc={test_metrics['two_models']['multiclass_acc']:.4f}, "
                              f"W-Acc={test_metrics['two_models']['multiclass_weighted_acc']:.4f}, "
                              f"Prec={test_metrics['two_models']['multiclass_precision']:.4f}, "
                              f"Rec={test_metrics['two_models']['multiclass_recall']:.4f}, "
                              f"F1={test_metrics['two_models']['multiclass_f1']:.4f}")
                        
                        # Save results
                        round_results = {
                            'round': round_num,
                            'client_metrics': client_metrics,
                            'average_validation_metrics': avg_metrics,
                            'global_validation_metrics': val_metrics,
                            'test_metrics': test_metrics
                        }
                        round_results = convert_keys_to_json_serializable(round_results)
                        with open(os.path.join(algorithm_folder, f'round_{round_num}_results.json'), 'w') as f:
                            json.dump(round_results, f, indent=4, cls=NumpyEncoder)
                        
                        # Save results to CSV
                        if results:
                            keys = ['client_id', 'num_clients', 'num_samples']
                            metric_keys = [
                                'binary_acc', 'binary_weighted_acc', 'binary_precision', 'binary_recall', 'binary_f1',
                                'fall_acc', 'fall_weighted_acc', 'fall_precision', 'fall_recall', 'fall_f1',
                                'non_fall_acc', 'non_fall_weighted_acc', 'non_fall_precision', 'non_fall_recall', 'non_fall_f1'
                            ]
                            with open(os.path.join(algorithm_folder, f'round_{round_num}_results.csv'), 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=keys + metric_keys)
                                writer.writeheader()
                                for result in results:
                                    row = {
                                        'client_id': result['client_id'],
                                        'num_clients': config.NUM_CLIENTS,
                                        'num_samples': result['num_samples']
                                    }
                                    row.update({k: result['metrics'].get(k, 0.0) for k in metric_keys})
                                    writer.writerow(row)
                            print(f"Results saved to {os.path.join(algorithm_folder, f'round_{round_num}_results.csv')}")