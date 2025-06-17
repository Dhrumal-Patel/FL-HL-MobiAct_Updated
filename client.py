import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from model import LSTMModel
from server import load_model_from_csv, save_model_to_csv

class FederatedClient:
    def __init__(self, client_id, train_dataset, test_dataset, config):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = config.DEVICE
        self.models = {
            'binary': LSTMModel(9, config.HIDDEN_SIZE_BINARY, config.NUM_LAYERS, 2).to(self.device),
            'fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.FALL_SCENARIOS)).to(self.device),
            'non_fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.NON_FALL_SCENARIOS)).to(self.device)
        }
        # print(f"[FederatedClient] Initialized client {client_id} with {len(train_dataset)} training samples")

    def load_global_model(self, model_save_folder):
        """Load global model parameters from CSV files."""
        for model_name in ['binary', 'fall', 'non_fall']:
            global_model_path = os.path.join(model_save_folder, f'global_{model_name}_params.csv')
            if os.path.exists(global_model_path):
                load_model_from_csv(self.models[model_name], global_model_path, self.device, self.config)
                # print(f"[FederatedClient] Loaded global {model_name} model for client {self.client_id}")
            else:
                print(f"[FederatedClient] Global {model_name} model file {global_model_path} not found, using local model")

    def train(self, model_save_folder, algorithm, epochs):
        """Train local models for the specified number of epochs."""
        criterion = nn.CrossEntropyLoss()
        optimizers = {
            'binary': torch.optim.Adam(self.models['binary'].parameters(), lr=self.config.LEARNING_RATE),
            'fall': torch.optim.Adam(self.models['fall'].parameters(), lr=self.config.LEARNING_RATE),
            'non_fall': torch.optim.Adam(self.models['non_fall'].parameters(), lr=self.config.LEARNING_RATE)
        }
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        for epoch in range(epochs):
            print(f"[FederatedClient] Training epoch {epoch+1}/{epochs} for client {self.client_id}")
            for model_name in self.models:
                self.models[model_name].train()
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    inputs, targets = batch
                    # print(f"[FederatedClient] Client {self.client_id}, Epoch {epoch+1}, Batch {batch_idx+1}: Inputs shape {inputs.shape}, Targets shape {targets.shape}")
                    
                    if targets.size(1) < 2:
                        print(f"[FederatedClient] Warning: Invalid targets shape {targets.shape} for client {self.client_id}, skipping batch")
                        continue
                    
                    inputs = inputs.to(self.device)
                    binary_targets, scenario_targets = targets[:, 0].long().to(self.device), targets[:, 1].long().to(self.device)
                    
                    # Train binary model
                    optimizers['binary'].zero_grad()
                    binary_out = self.models['binary'](inputs)
                    binary_loss = criterion(binary_out, binary_targets)
                    binary_loss.backward()
                    optimizers['binary'].step()
                    
                    # Train fall model for fall samples
                    fall_mask = binary_targets == 0
                    if fall_mask.any():
                        optimizers['fall'].zero_grad()
                        fall_out = self.models['fall'](inputs[fall_mask])
                        fall_loss = criterion(fall_out, scenario_targets[fall_mask])
                        fall_loss.backward()
                        optimizers['fall'].step()
                    
                    # Train non-fall model for non-fall samples
                    non_fall_mask = binary_targets == 1
                    if non_fall_mask.any():
                        optimizers['non_fall'].zero_grad()
                        non_fall_out = self.models['non_fall'](inputs[non_fall_mask])
                        non_fall_loss = criterion(non_fall_out, scenario_targets[non_fall_mask])
                        non_fall_loss.backward()
                        optimizers['non_fall'].step()
                
                except Exception as e:
                    print(f"[FederatedClient] Error processing batch {batch_idx+1} for client {self.client_id}: {str(e)}")
                    continue
        
        # Save local models
        self.save_model(model_save_folder)
        
        # Evaluate and return metrics
        metrics = self.evaluate()
        num_samples = len(self.train_dataset)
        return metrics, num_samples

    def evaluate(self):
        """Evaluate local models on test dataset."""
        criterion = nn.CrossEntropyLoss()
        for model_name in self.models:
            self.models[model_name].eval()
        
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        metrics = {
            'binary_acc': 0.0, 'binary_weighted_acc': 0.0, 'binary_precision': 0.0, 'binary_recall': 0.0, 'binary_f1': 0.0,
            'fall_acc': 0.0, 'fall_weighted_acc': 0.0, 'fall_precision': 0.0, 'fall_recall': 0.0, 'fall_f1': 0.0,
            'non_fall_acc': 0.0, 'non_fall_weighted_acc': 0.0, 'non_fall_precision': 0.0, 'non_fall_recall': 0.0, 'non_fall_f1': 0.0
        }
        
        binary_correct, fall_correct, non_fall_correct = 0, 0, 0
        total, fall_total, non_fall_total = 0, 0, 0
        binary_preds, binary_targets_list = [], []
        fall_preds, fall_targets_list = [], []
        non_fall_preds, non_fall_targets_list = [], []
        binary_class_counts, fall_class_counts, non_fall_class_counts = {}, {}, {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    inputs, targets = batch
                    # print(f"[FederatedClient] Client {self.client_id}, Eval Batch {batch_idx+1}: Inputs shape {inputs.shape}, Targets shape {targets.shape}")
                    
                    if targets.size(1) < 2:
                        print(f"[FederatedClient] Warning: Invalid targets shape {targets.shape} in evaluation for client {self.client_id}, skipping batch")
                        continue
                    
                    inputs = inputs.to(self.device)
                    binary_targets_batch, scenario_targets_batch = targets[:, 0].long().to(self.device), targets[:, 1].long().to(self.device)
                    
                    # Binary model
                    binary_out = self.models['binary'](inputs)
                    _, binary_pred = torch.max(binary_out, 1)
                    binary_correct += (binary_pred == binary_targets_batch).sum().item()
                    binary_preds.extend(binary_pred.cpu().tolist())
                    binary_targets_list.extend(binary_targets_batch.cpu().tolist())
                    for target in binary_targets_batch.cpu().numpy():
                        binary_class_counts[target] = binary_class_counts.get(target, 0) + 1
                    total += len(binary_targets_batch)
                    
                    # Fall model
                    fall_mask = binary_targets_batch == 0
                    if fall_mask.any():
                        fall_out = self.models['fall'](inputs[fall_mask])
                        _, fall_pred = torch.max(fall_out, 1)
                        fall_correct += (fall_pred == scenario_targets_batch[fall_mask]).sum().item()
                        fall_preds.extend(fall_pred.cpu().tolist())
                        fall_targets_list.extend(scenario_targets_batch[fall_mask].cpu().tolist())
                        for target in scenario_targets_batch[fall_mask].cpu().numpy():
                            fall_class_counts[target] = fall_class_counts.get(target, 0) + 1
                        fall_total += fall_mask.sum().item()
                    
                    # Non-fall model
                    non_fall_mask = binary_targets_batch == 1
                    if non_fall_mask.any():
                        non_fall_out = self.models['non_fall'](inputs[non_fall_mask])
                        _, non_fall_pred = torch.max(non_fall_out, 1)
                        non_fall_correct += (non_fall_pred == scenario_targets_batch[non_fall_mask]).sum().item()
                        non_fall_preds.extend(non_fall_pred.cpu().tolist())
                        non_fall_targets_list.extend(scenario_targets_batch[non_fall_mask].cpu().tolist())
                        for target in scenario_targets_batch[non_fall_mask].cpu().numpy():
                            non_fall_class_counts[target] = non_fall_class_counts.get(target, 0) + 1
                        non_fall_total += non_fall_mask.sum().item()
                
                except Exception as e:
                    print(f"[FederatedClient] Error evaluating batch {batch_idx+1} for client {self.client_id}: {str(e)}")
                    continue
        
        # Compute accuracies
        metrics['binary_acc'] = binary_correct / max(1, total)
        metrics['fall_acc'] = fall_correct / max(1, fall_total) if fall_total > 0 else 0.0
        metrics['non_fall_acc'] = non_fall_correct / max(1, non_fall_total) if non_fall_total > 0 else 0.0
        
        # Compute weighted accuracies
        if binary_class_counts:
            binary_weights = np.array([binary_class_counts.get(i, 0) for i in range(2)])
            binary_weights = binary_weights / binary_weights.sum() if binary_weights.sum() > 0 else np.ones(2) / 2
            binary_correct_weighted = sum(
                sum(1 for pred, target in zip(binary_preds, binary_targets_list) if pred == target == i) * binary_weights[i]
                for i in range(2)
            )
            metrics['binary_weighted_acc'] = binary_correct_weighted / max(1, total)
        
        if fall_class_counts:
            fall_weights = np.array([fall_class_counts.get(i, 0) for i in range(len(self.config.FALL_SCENARIOS))])
            fall_weights = fall_weights / fall_weights.sum() if fall_weights.sum() > 0 else np.ones(len(self.config.FALL_SCENARIOS)) / len(self.config.FALL_SCENARIOS)
            fall_correct_weighted = sum(
                sum(1 for pred, target in zip(fall_preds, fall_targets_list) if pred == target == i) * fall_weights[i]
                for i in range(len(self.config.FALL_SCENARIOS))
            )
            metrics['fall_weighted_acc'] = fall_correct_weighted / max(1, fall_total) if fall_total > 0 else 0.0
        
        if non_fall_class_counts:
            non_fall_weights = np.array([non_fall_class_counts.get(i, 0) for i in range(len(self.config.NON_FALL_SCENARIOS))])
            non_fall_weights = non_fall_weights / non_fall_weights.sum() if non_fall_weights.sum() > 0 else np.ones(len(self.config.NON_FALL_SCENARIOS)) / len(self.config.NON_FALL_SCENARIOS)
            non_fall_correct_weighted = sum(
                sum(1 for pred, target in zip(non_fall_preds, non_fall_targets_list) if pred == target == i) * non_fall_weights[i]
                for i in range(len(self.config.NON_FALL_SCENARIOS))
            )
            metrics['non_fall_weighted_acc'] = non_fall_correct_weighted / max(1, non_fall_total) if non_fall_total > 0 else 0.0
        
        # Compute precision, recall, F1
        if binary_targets_list:
            binary_prf = precision_recall_fscore_support(binary_targets_list, binary_preds, average='weighted', zero_division=0)
            metrics['binary_precision'], metrics['binary_recall'], metrics['binary_f1'] = binary_prf[0], binary_prf[1], binary_prf[2]
        
        if fall_targets_list:
            fall_prf = precision_recall_fscore_support(fall_targets_list, fall_preds, average='weighted', zero_division=0)
            metrics['fall_precision'], metrics['fall_recall'], metrics['fall_f1'] = fall_prf[0], fall_prf[1], fall_prf[2]
        
        if non_fall_targets_list:
            non_fall_prf = precision_recall_fscore_support(non_fall_targets_list, non_fall_preds, average='weighted', zero_division=0)
            metrics['non_fall_precision'], metrics['non_fall_recall'], metrics['non_fall_f1'] = non_fall_prf[0], non_fall_prf[1], non_fall_prf[2]
        
        return metrics

    def save_model(self, model_save_folder):
        """Save local model parameters to CSV files."""
        client_folder = os.path.join(model_save_folder, f'client_{self.client_id}')
        os.makedirs(client_folder, exist_ok=True)
        for model_name in self.models:
            save_model_to_csv(
                self.models[model_name],
                os.path.join(client_folder, f'{model_name}_params.csv'),
                self.config
            )
            # print(f"[FederatedClient] Saved {model_name} model for client {self.client_id}")