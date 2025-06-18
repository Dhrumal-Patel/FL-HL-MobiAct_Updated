import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import LSTMModel
from server import save_model_to_csv, load_model_from_csv
from sklearn.metrics import precision_recall_fscore_support
import os

class FederatedClient:
    def __init__(self, client_id, train_data, test_data, config):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.device = config.DEVICE
        self.models = {
            'binary': LSTMModel(9, config.HIDDEN_SIZE_BINARY, num_layers=config.NUM_LAYERS, num_classes=2).to(self.device),
            'fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, num_layers=config.NUM_LAYERS, num_classes=len(config.FALL_SCENARIOS)).to(self.device),
            'non_fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, num_layers=config.NUM_LAYERS, num_classes=len(config.NON_FALL_SCENARIOS)).to(self.device)
        }
        self.criterion = nn.CrossEntropyLoss()

    def load_global_model(self, model_path):
        for model_name in self.models:
            load_model_from_csv(
                self.models[model_name],
                f"{model_path}/global_{model_name}_params.csv",
                self.device,
                self.config
            )

    def train(self, model_path, algorithm, epochs):
        # Validate train_data
        if len(self.train_data) == 0:
            print(f"Client {self.client_id}: Empty train_data, skipping training")
            metrics = {
                'binary_acc': 0.0, 'fall_acc': 0.0, 'non_fall_acc': 0.0,
                'binary_weighted_acc': 0.0, 'fall_weighted_acc': 0.0, 'non_fall_weighted_acc': 0.0,
                'binary_precision': 0.0, 'binary_recall': 0.0, 'binary_f1': 0.0,
                'fall_precision': 0.0, 'fall_recall': 0.0, 'fall_f1': 0.0,
                'non_fall_precision': 0.0, 'non_fall_recall': 0.0, 'non_fall_f1': 0.0
            }
            return metrics, 0, 0, 0

        # print(f"Client {self.client_id}: train_data size = {len(self.train_data)}")
        # Debug: Inspect first sample
        try:
            inputs, targets = self.train_data[0]
            # print(f"Client {self.client_id}: Sample inputs shape = {inputs.shape}, targets shape = {targets.shape}")
        except Exception as e:
            print(f"Client {self.client_id}: Error accessing first sample: {e}")

        for model in self.models.values():
            model.train()

        optimizer_binary = torch.optim.Adam(self.models['binary'].parameters(), lr=self.config.LEARNING_RATE)
        optimizer_fall = torch.optim.Adam(self.models['fall'].parameters(), lr=self.config.LEARNING_RATE)
        optimizer_non_fall = torch.optim.Adam(self.models['non_fall'].parameters(), lr=self.config.LEARNING_RATE)

        train_loader = DataLoader(self.train_data, batch_size=self.config.BATCH_SIZE, shuffle=True)

        # Debug: Check DataLoader batches
        try:
            batch_count = 0
            for inputs, targets in train_loader:
                batch_count += 1
            # print(f"Client {self.client_id}: DataLoader yielded {batch_count} batches")
            if batch_count == 0:
                print(f"Client {self.client_id}: DataLoader yielded no batches")
                metrics = {
                    'binary_acc': 0.0, 'fall_acc': 0.0, 'non_fall_acc': 0.0,
                    'binary_weighted_acc': 0.0, 'fall_weighted_acc': 0.0, 'non_fall_weighted_acc': 0.0,
                    'binary_precision': 0.0, 'binary_recall': 0.0, 'binary_f1': 0.0,
                    'fall_precision': 0.0, 'fall_recall': 0.0, 'fall_f1': 0.0,
                    'non_fall_precision': 0.0, 'non_fall_recall': 0.0, 'non_fall_f1': 0.0
                }
                return metrics, len(self.train_data), 0, 0
        except Exception as e:
            print(f"Client {self.client_id}: DataLoader iteration error: {e}")
            metrics = {
                'binary_acc': 0.0, 'fall_acc': 0.0, 'non_fall_acc': 0.0,
                'binary_weighted_acc': 0.0, 'fall_weighted_acc': 0.0, 'non_fall_weighted_acc': 0.0,
                'binary_precision': 0.0, 'binary_recall': 0.0, 'binary_f1': 0.0,
                'fall_precision': 0.0, 'fall_recall': 0.0, 'fall_f1': 0.0,
                'non_fall_precision': 0.0, 'non_fall_recall': 0.0, 'non_fall_f1': 0.0
            }
            return metrics, len(self.train_data), 0, 0

        # Create mapping for fall and non-fall scenarios to model indices
        fall_scenario_map = {scenario: idx for idx, scenario in enumerate(sorted(self.config.FALL_SCENARIOS))}
        non_fall_scenario_map = {scenario: idx for idx, scenario in enumerate(sorted(self.config.NON_FALL_SCENARIOS))}

        for epoch in range(epochs):
            print(f"Client {self.client_id}: Starting epoch {epoch+1}/{epochs}")
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                binary_targets, multi_targets = targets[:, 0], targets[:, 1]

                # Map multi_targets to fall and non-fall indices
                fall_targets = torch.tensor([fall_scenario_map.get(t.item(), 0) for t in multi_targets], dtype=torch.long).to(self.device)
                non_fall_targets = torch.tensor([non_fall_scenario_map.get(t.item(), 0) for t in multi_targets], dtype=torch.long).to(self.device)

                optimizer_binary.zero_grad()
                binary_out = self.models['binary'](inputs)
                binary_loss = self.criterion(binary_out, binary_targets)
                binary_loss.backward()
                optimizer_binary.step()

                fall_mask = binary_targets == 0
                if fall_mask.any():
                    optimizer_fall.zero_grad()
                    fall_out = self.models['fall'](inputs[fall_mask])
                    fall_loss = self.criterion(fall_out, fall_targets[fall_mask])
                    fall_loss.backward()
                    optimizer_fall.step()

                non_fall_mask = binary_targets == 1
                if non_fall_mask.any():
                    optimizer_non_fall.zero_grad()
                    non_fall_out = self.models['non_fall'](inputs[non_fall_mask])
                    non_fall_loss = self.criterion(non_fall_out, non_fall_targets[non_fall_mask])
                    non_fall_loss.backward()
                    optimizer_non_fall.step()
            # print(f"Client {self.client_id}: Completed epoch {epoch+1}/{epochs}")

        metrics = self.evaluate()
        total_samples = len(self.train_data)
        fall_samples = sum(1 for _, target in self.train_data if target[0] == 0)
        non_fall_samples = total_samples - fall_samples

        print(f"Client {self.client_id}: Total samples={total_samples}, Fall samples={fall_samples}, Non-Fall samples={non_fall_samples}")

        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        for model_name in self.models:
            try:
                save_model_to_csv(
                    self.models[model_name],
                    f"{model_path}/client_{self.client_id}_{model_name}_params.csv",
                    self.config
                )
                # print(f"Client {self.client_id}: Model saved to {model_path}/client_{self.client_id}_{model_name}_params.csv")
            except Exception as e:
                print(f"Client {self.client_id}: Error saving model {model_name}: {e}")

        return metrics, total_samples, fall_samples, non_fall_samples

    def evaluate(self):
        for model in self.models.values():
            model.eval()

        test_loader = DataLoader(self.test_data, batch_size=self.config.BATCH_SIZE, shuffle=False)
        binary_correct, fall_correct, non_fall_correct = 0, 0, 0
        total, fall_total, non_fall_total = 0, 0, 0
        binary_preds, binary_targets = [], []
        fall_preds, fall_targets = [], []
        non_fall_preds, non_fall_targets = [], []
        binary_class_counts = {}
        fall_class_counts = {}
        non_fall_class_counts = {}

        # Create mapping for fall and non-fall scenarios to model indices
        fall_scenario_map = {scenario: idx for idx, scenario in enumerate(sorted(self.config.FALL_SCENARIOS))}
        non_fall_scenario_map = {scenario: idx for idx, scenario in enumerate(sorted(self.config.NON_FALL_SCENARIOS))}

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                binary_targets_batch, multi_targets_batch = targets[:, 0], targets[:, 1]

                # Map multi_targets to fall and non-fall indices
                fall_targets_batch = torch.tensor([fall_scenario_map.get(t.item(), 0) for t in multi_targets_batch], dtype=torch.long).to(self.device)
                non_fall_targets_batch = torch.tensor([non_fall_scenario_map.get(t.item(), 0) for t in multi_targets_batch], dtype=torch.long).to(self.device)

                binary_out = self.models['binary'](inputs)
                _, binary_pred = torch.max(binary_out, 1)
                binary_correct += (binary_pred == binary_targets_batch).sum().item()
                binary_preds.extend(binary_pred.tolist())
                binary_targets.extend(binary_targets_batch.tolist())

                for target in binary_targets_batch.cpu().numpy():
                    binary_class_counts[int(target)] = binary_class_counts.get(int(target), 0) + 1

                fall_mask = binary_targets_batch == 0
                if fall_mask.any():
                    fall_out = self.models['fall'](inputs[fall_mask])
                    _, fall_pred = torch.max(fall_out, 1)
                    fall_correct += (fall_pred == fall_targets_batch[fall_mask]).sum().item()
                    fall_total += fall_mask.sum().item()
                    fall_preds.extend(fall_pred.tolist())
                    fall_targets.extend(fall_targets_batch[fall_mask].tolist())
                    for target in fall_targets_batch[fall_mask].cpu().numpy():
                        fall_class_counts[int(target)] = fall_class_counts.get(int(target), 0) + 1

                non_fall_mask = binary_targets_batch == 1
                if non_fall_mask.any():
                    non_fall_out = self.models['non_fall'](inputs[non_fall_mask])
                    _, non_fall_pred = torch.max(non_fall_out, 1)
                    non_fall_correct += (non_fall_pred == non_fall_targets_batch[non_fall_mask]).sum().item()
                    non_fall_total += non_fall_mask.sum().item()
                    non_fall_preds.extend(non_fall_pred.tolist())
                    non_fall_targets.extend(non_fall_targets_batch[non_fall_mask].tolist())
                    for target in non_fall_targets_batch[non_fall_mask].cpu().numpy():
                        non_fall_class_counts[int(target)] = non_fall_class_counts.get(int(target), 0) + 1

                total += len(targets)

        metrics = {
            'binary_acc': binary_correct / max(1, total),
            'fall_acc': fall_correct / max(1, fall_total),
            'non_fall_acc': non_fall_correct / max(1, non_fall_total),
            'binary_weighted_acc': 0.0,
            'fall_weighted_acc': 0.0,
            'non_fall_weighted_acc': 0.0,
            'binary_precision': 0.0,
            'binary_recall': 0.0,
            'binary_f1': 0.0,
            'fall_precision': 0.0,
            'fall_recall': 0.0,
            'fall_f1': 0.0,
            'non_fall_precision': 0.0,
            'non_fall_recall': 0.0,
            'non_fall_f1': 0.0
        }

        if binary_class_counts:
            binary_weights = np.array([binary_class_counts.get(i, 0) for i in range(2)])
            binary_weights = binary_weights / binary_weights.sum() if binary_weights.sum() > 0 else np.ones(2) / 2
            binary_correct_weighted = 0
            for i in range(2):
                correct_count = sum(1 for pred, target in zip(binary_preds, binary_targets) if pred == target == i)
                binary_correct_weighted += correct_count * binary_weights[i]
            metrics['binary_weighted_acc'] = binary_correct_weighted / max(1, total)

        if fall_class_counts:
            fall_weights = np.array([fall_class_counts.get(i, 0) for i in range(len(self.config.FALL_SCENARIOS))])
            fall_weights = fall_weights / fall_weights.sum() if fall_weights.sum() > 0 else np.ones(len(self.config.FALL_SCENARIOS)) / len(self.config.FALL_SCENARIOS)
            fall_correct_weighted = 0
            for i in range(len(self.config.FALL_SCENARIOS)):
                correct_count = sum(1 for pred, target in zip(fall_preds, fall_targets) if pred == target == i)
                fall_correct_weighted += correct_count * fall_weights[i]
            metrics['fall_weighted_acc'] = fall_correct_weighted / max(1, fall_total)

        if non_fall_class_counts:
            non_fall_weights = np.array([non_fall_class_counts.get(i, 0) for i in range(len(self.config.NON_FALL_SCENARIOS))])
            non_fall_weights = non_fall_weights / non_fall_weights.sum() if non_fall_weights.sum() > 0 else np.ones(len(self.config.NON_FALL_SCENARIOS)) / len(self.config.NON_FALL_SCENARIOS)
            non_fall_correct_weighted = 0
            for i in range(len(self.config.NON_FALL_SCENARIOS)):
                correct_count = sum(1 for pred, target in zip(non_fall_preds, non_fall_targets) if pred == target == i)
                non_fall_correct_weighted += correct_count * non_fall_weights[i]
            metrics['non_fall_weighted_acc'] = non_fall_correct_weighted / max(1, non_fall_total)

        if binary_targets:
            binary_prf = precision_recall_fscore_support(binary_targets, binary_preds, average='weighted', zero_division=0)
            metrics['binary_precision'] = binary_prf[0]
            metrics['binary_recall'] = binary_prf[1]
            metrics['binary_f1'] = binary_prf[2]

        if fall_targets:
            fall_prf = precision_recall_fscore_support(fall_targets, fall_preds, average='weighted', zero_division=0)
            metrics['fall_precision'] = fall_prf[0]
            metrics['fall_recall'] = fall_prf[1]
            metrics['fall_f1'] = fall_prf[2]

        if non_fall_targets:
            non_fall_prf = precision_recall_fscore_support(non_fall_targets, non_fall_preds, average='weighted', zero_division=0)
            metrics['non_fall_precision'] = non_fall_prf[0]
            metrics['non_fall_recall'] = non_fall_prf[1]
            metrics['non_fall_f1'] = non_fall_prf[2]

        return metrics
