import os
import csv
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from client import FederatedClient
from model import LSTMModel
from server import load_model_from_csv, weighted_fedavg
from dataset import prepare_federated_data
from config import Config
import torch.nn as nn

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

def main():
    config = Config()
    algorithms = ['weighted_fedavg']
    overlaps = [0.0]
    num_clients_list = [3]

    for overlap in overlaps:
        for num_clients in num_clients_list:
            for algorithm in algorithms:
                print(f"\n=== Running {algorithm.upper()} with Overlap = {overlap}, Num Clients = {num_clients} ===")
                
                # Create directories
                model_path = f"{config.MODEL_SAVE_FOLDER}/overlap_{overlap}_num_clients_{num_clients}"
                results_dir = f"{config.SAVE_FOLDER}/overlap_{overlap}_num_clients_{num_clients}/{algorithm}"
                os.makedirs(model_path, exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)

                # Prepare data
                client_datasets, val_dataset, test_dataset = prepare_federated_data(config)
                print(f"Prepared {len(client_datasets)} client datasets, validation dataset, and test dataset.")

                clients = [
                    FederatedClient(i, client_datasets[i], test_dataset, config)
                    for i in range(num_clients)
                ]

                # Define CSV headers based on all metrics
                metric_keys = [
                    'binary_acc', 'binary_weighted_acc', 'binary_precision', 'binary_recall', 'binary_f1',
                    'fall_acc', 'fall_weighted_acc', 'fall_precision', 'fall_recall', 'fall_f1',
                    'non_fall_acc', 'non_fall_weighted_acc', 'non_fall_precision', 'non_fall_recall', 'non_fall_f1'
                ]
                headers = ['Round', 'Client/Global', 'Num_Clients', 'Num_Samples']
                headers.extend([f'Val_{key}' for key in metric_keys])
                headers.extend([f'Test_{key}' for key in metric_keys])

                for round_num in range(1, config.COMMUNICATION_ROUNDS + 1):
                    print(f"\n=== {algorithm.upper()} Round {round_num}/{config.COMMUNICATION_ROUNDS} ===")
                    
                    client_metrics = []
                    total_samples_list, fall_samples_list, non_fall_samples_list = [], [], []

                    # Train clients and collect metrics
                    for client in clients:
                        print(f"Training client {client.client_id}")
                        try:
                            metrics, total_samples, fall_samples, non_fall_samples = client.train(
                                model_path, algorithm, config.CLIENT_EPOCHS
                            )
                            print(f"Client {client.client_id}: Training completed, metrics: {metrics}")
                            client_metrics.append(metrics)
                            total_samples_list.append(total_samples)
                            fall_samples_list.append(fall_samples)
                            non_fall_samples_list.append(non_fall_samples)

                            # Write client metrics to CSV
                            row = [round_num, f'Client_{client.client_id}', num_clients, total_samples]
                            for key in metric_keys:
                                val_value = metrics.get(key, 0.0)  # Use training metrics as validation placeholder
                                test_value = 0.0  # Placeholder for test metrics
                                row.extend([val_value, test_value])
                            with open(f"{results_dir}/results.csv", 'a', newline='') as f:
                                writer = csv.writer(f)
                                if round_num == 1 and client.client_id == 0:  # Write header only once
                                    writer.writerow(headers)
                                writer.writerow(row)
                        except Exception as e:
                            print(f"Client {client.client_id}: Training failed: {e}")

                    # Aggregate and evaluate global model
                    global_model = weighted_fedavg(
                        clients, model_path, config, total_samples_list, fall_samples_list, non_fall_samples_list
                    )

                    # Evaluate global model on validation and test sets
                    from torch.utils.data import DataLoader
                    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
                    val_metrics = evaluate_global_model(None, val_loader, config, model_name='validation')
                    test_metrics = evaluate_global_model(None, test_loader, config, model_name='test')

                    # Write global metrics to CSV
                    global_row = [round_num, 'Global', num_clients, sum(total_samples_list)]
                    for key in metric_keys:
                        val_value = val_metrics['three_models'].get(key, 0.0)
                        test_value = test_metrics['three_models'].get(key, 0.0)
                        global_row.extend([val_value, test_value])
                    with open(f"{results_dir}/results.csv", 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(global_row)

                    # Prepare JSON data
                    round_results = {
                        'round': round_num,
                        'client_metrics': {f'Client_{i}': convert_keys_to_json_serializable(metrics) for i, metrics in enumerate(client_metrics)},
                        'global_validation_metrics': val_metrics,
                        'global_test_metrics': test_metrics
                    }
                    with open(f"{results_dir}/round_{round_num}_results.json", 'w') as f:
                        json.dump(round_results, f, indent=4, cls=NumpyEncoder)

                    print(f"Results saved to {results_dir}/round_{round_num}_results.json and {results_dir}/results.csv")

if __name__ == "__main__":
    main()
