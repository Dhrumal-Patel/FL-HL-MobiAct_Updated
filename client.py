import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import LSTMModel, save_model_to_csv
from sklearn.metrics import precision_recall_fscore_support

class FederatedClient:
    def __init__(self, client_id, train_dataset, val_dataset, config):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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
    
    def load_global_model(self, model_save_folder):
        from server import load_model_from_csv
        for model_name in self.models:
            model_path = f"{model_save_folder}/global_{model_name}_params.csv"
            load_model_from_csv(self.models[model_name], model_path, self.device, self.config)
    
    def evaluate(self, loader, model_name='val'):
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        preds = []
        targets_list = []
        class_counts = {}
        
        self.models[model_name].eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if model_name == 'binary':
                    target_col = 0
                else:
                    target_col = 1
                
                outputs = self.models[model_name](inputs).to(self.device)
                loss = criterion(outputs, targets[:, target_col])
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets[:, target_col]).sum().item()
                total += len(targets)
                preds.extend(predicted.tolist())
                targets_list.extend(targets[:, target_col].tolist())
                
                for target in targets[:, target_col].cpu().numpy():
                    class_counts[int(target)] = class_counts.get(int(target), 0) + 1
        
        accuracy = correct / max(1, total)
        weighted_acc = 0.0
        if class_counts:
            weights = np.array([class_counts.get(i, 0) for i in range(max(class_counts.keys()) + 1)])
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
            for i in range(len(weights)):
                correct_count = sum(1 for pred, target in zip(preds, targets_list) if pred == target == i)
                weighted_acc += correct_count * weights[i]
            weighted_acc /= max(1, total)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_list, preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': total_loss / max(1, len(loader)),
            'accuracy': accuracy,
            'weighted_accuracy': weighted_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, model_save_folder, algorithm, epochs):
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizers = {
            'binary': torch.optim.Adam(self.models['binary'].parameters(), lr=self.config.LEARNING_RATE),
            'fall': torch.optim.Adam(self.models['fall'].parameters(), lr=self.config.LEARNING_RATE),
            'non_fall': torch.optim.Adam(self.models['non_fall'].parameters(), lr=self.config.LEARNING_RATE)
        }
        
        # Count fall and non-fall samples
        fall_samples = 0
        non_fall_samples = 0
        for _, targets in train_loader:
            binary_targets = targets[:, 0].cpu().numpy()
            fall_samples += np.sum(binary_targets == 0)
            non_fall_samples += np.sum(binary_targets == 1)
        
        total_samples = len(self.train_dataset)
        print(f"Client {self.client_id}: Total samples={total_samples}, Fall samples={fall_samples}, Non-Fall samples={non_fall_samples}")
        
        metrics = {
            'binary_loss': 0.0, 'binary_acc': 0.0, 'binary_weighted_acc': 0.0,
            'binary_precision': 0.0, 'binary_recall': 0.0, 'binary_f1': 0.0,
            'fall_loss': 0.0, 'fall_acc': 0.0, 'fall_weighted_acc': 0.0,
            'fall_precision': 0.0, 'fall_recall': 0.0, 'fall_f1': 0.0,
            'non_fall_loss': 0.0, 'non_fall_acc': 0.0, 'non_fall_weighted_acc': 0.0,
            'non_fall_precision': 0.0, 'non_fall_recall': 0.0, 'non_fall_f1': 0.0
        }
        
        for model_name in self.models:
            self.models[model_name].train()
        
        for epoch in range(epochs):
            total_loss = {'binary': 0.0, 'fall': 0.0, 'non_fall': 0.0}
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                binary_targets, multi_targets = targets[:, 0], targets[:, 1]
                
                # Binary model
                optimizers['binary'].zero_grad()
                binary_out = self.models['binary'](inputs).to(self.device)
                binary_loss = criterion(binary_out, binary_targets)
                total_loss['binary'] += binary_loss.item()
                binary_loss.backward()
                optimizers['binary'].step()
                
                # Fall model
                fall_mask = binary_targets == 0
                if fall_mask.any():
                    optimizers['fall'].zero_grad()
                    fall_out = self.models['fall'](inputs[fall_mask]).to(self.device)
                    fall_loss = criterion(fall_out, multi_targets[fall_mask])
                    total_loss['fall'] += fall_loss.item()
                    fall_loss.backward()
                    optimizers['fall'].step()
                
                # Non-fall model
                non_fall_mask = binary_targets == 1
                if non_fall_mask.any():
                    optimizers['non_fall'].zero_grad()
                    non_fall_out = self.models['non_fall'](inputs[non_fall_mask]).to(self.device)
                    non_fall_loss = criterion(non_fall_out, multi_targets[non_fall_mask])
                    total_loss['non_fall'] += non_fall_loss.item()
                    non_fall_loss.backward()
                    optimizers['non_fall'].step()
            
            # Evaluate after each epoch
            train_metrics = self.evaluate(train_loader, 'binary')
            metrics['binary_loss'] = total_loss['binary'] / len(train_loader)
            metrics['binary_acc'] = train_metrics['accuracy']
            metrics['binary_weighted_acc'] = train_metrics['weighted_accuracy']
            metrics['binary_precision'] = train_metrics['precision']
            metrics['binary_recall'] = train_metrics['recall']
            metrics['binary_f1'] = train_metrics['f1']
            
            if fall_samples > 0:
                train_metrics = self.evaluate(train_loader, 'fall')
                metrics['fall_loss'] = total_loss['fall'] / max(1, sum(binary_targets == 0 for _, targets in train_loader))
                metrics['fall_acc'] = train_metrics['accuracy']
                metrics['fall_weighted_acc'] = train_metrics['weighted_accuracy']
                metrics['fall_precision'] = train_metrics['precision']
                metrics['fall_recall'] = train_metrics['recall']
                metrics['fall_f1'] = train_metrics['f1']
            
            if non_fall_samples > 0:
                train_metrics = self.evaluate(train_loader, 'non_fall')
                metrics['non_fall_loss'] = total_loss['non_fall'] / max(1, sum(binary_targets == 1 for _, targets in train_loader))
                metrics['non_fall_acc'] = train_metrics['accuracy']
                metrics['non_fall_weighted_acc'] = train_metrics['weighted_accuracy']
                metrics['non_fall_precision'] = train_metrics['precision']
                metrics['non_fall_recall'] = train_metrics['recall']
                metrics['non_fall_f1'] = train_metrics['f1']
        
        # Save models
        for model_name in self.models:
            save_model_to_csv(
                self.models[model_name],
                f"{model_save_folder}/client_{self.client_id}/{model_name}_params.csv",
                self.config
            )
        
        return metrics, total_samples, fall_samples, non_fall_samples
