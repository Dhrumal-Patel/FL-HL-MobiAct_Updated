import os
import torch
import numpy as np
from model import LSTMModel
import pandas as pd

def save_model_to_csv(model, file_path, config):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    param_dict = {name: param.cpu().detach().numpy().flatten() for name, param in model.named_parameters()}
    max_length = max(len(arr) for arr in param_dict.values())
    for key in param_dict:
        param_dict[key] = np.pad(param_dict[key], (0, max_length - len(param_dict[key])), mode='constant')
    df = pd.DataFrame(param_dict)
    df.to_csv(file_path, index=False)

def load_model_from_csv(model, file_path, device, config):
    if not os.path.exists(file_path):
        print(f"Model file {file_path} does not exist. Initializing with default weights.")
        return
    df = pd.read_csv(file_path)
    param_dict = {}
    for key in df.columns:
        param_dict[key] = df[key].values
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if name in param_dict:
            param_data = param_dict[name][:param.numel()].reshape(param.shape)
            state_dict[name].copy_(torch.tensor(param_data, device=device))
    model.load_state_dict(state_dict)

def weighted_fedavg(clients, model_path, config, total_samples_list, fall_samples_list, non_fall_samples_list):
    device = config.DEVICE
    models = {
        'binary': LSTMModel(9, config.HIDDEN_SIZE_BINARY, num_layers=config.NUM_LAYERS, num_classes=2).to(device),
        'fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, num_layers=config.NUM_LAYERS, num_classes=len(config.FALL_SCENARIOS)).to(device),
        'non_fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, num_layers=config.NUM_LAYERS, num_classes=len(config.NON_FALL_SCENARIOS)).to(device)
    }

    valid_clients = []
    total_samples_sum = sum(total_samples_list)
    fall_samples_sum = sum(fall_samples_list)
    non_fall_samples_sum = sum(non_fall_samples_list)

    for i, client in enumerate(clients):
        binary_file = f"{model_path}/client_{i}_binary_params.csv"
        fall_file = f"{model_path}/client_{i}_fall_params.csv"
        non_fall_file = f"{model_path}/client_{i}_non_fall_params.csv"
        if os.path.exists(binary_file) and os.path.exists(fall_file) and os.path.exists(non_fall_file):
            valid_clients.append(i)
        else:
            print(f"Client {i} model directory does not exist: binary={os.path.exists(binary_file)}, fall={os.path.exists(fall_file)}, non_fall={os.path.exists(non_fall_file)}")

    if not valid_clients:
        print("No valid clients for aggregation")
        for model_name in models:
            save_model_to_csv(models[model_name], f"{model_path}/global_{model_name}_params.csv", config)
        return {'binary_acc': 0.0, 'fall_acc': 0.0, 'non_fall_acc': 0.0}

    global_params = {model_name: {name: torch.zeros_like(param, device=device) for name, param in models[model_name].named_parameters()} for model_name in models}

    for model_name in models:
        for i in valid_clients:
            client_model = LSTMModel(
                9,
                config.HIDDEN_SIZE_BINARY if model_name == 'binary' else config.HIDDEN_SIZE_MULTICLASS,
                num_layers=config.NUM_LAYERS,
                num_classes=2 if model_name == 'binary' else len(config.FALL_SCENARIOS if model_name == 'fall' else config.NON_FALL_SCENARIOS)
            ).to(device)
            load_model_from_csv(client_model, f"{model_path}/client_{i}_{model_name}_params.csv", device, config)
            weight = total_samples_list[i] / total_samples_sum if model_name == 'binary' else (
                fall_samples_list[i] / fall_samples_sum if model_name == 'fall' else non_fall_samples_list[i] / non_fall_samples_sum
            )
            for name, param in client_model.named_parameters():
                global_params[model_name][name] += weight * param

    for model_name in models:
        models[model_name].load_state_dict(global_params[model_name])
        save_model_to_csv(models[model_name], f"{model_path}/global_{model_name}_params.csv", config)
        norm = sum(torch.norm(param).item() for param in models[model_name].parameters())
        # print(f"[Debug] Round {round_num}, {model_name} norm after aggregation: {norm:.4f}")

    return {'binary_acc': 0.0, 'fall_acc': 0.0, 'non_fall_acc': 0.0}
