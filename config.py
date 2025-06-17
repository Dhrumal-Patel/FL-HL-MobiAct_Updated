# config.py
import os
import torch
import csv

# Set a safe CSV field size limit
csv.field_size_limit(2147483647)

# Configuration
class Config:
    def __init__(self, overlap=0.0, num_clients=4):
        self.OVERLAP = overlap
        self.NUM_CLIENTS = num_clients
        self.SAVE_FOLDER = os.path.join(os.getcwd(), f'federated_results/overlap_{overlap}_num_clients_{num_clients}')
        self.MODEL_SAVE_FOLDER = os.path.join(os.getcwd(), f'model_params/overlap_{overlap}_num_clients_{num_clients}')
        self.DATA_FILE = 'E:/DA-IICT/SEM-2/Minor_Project/Final Paper/Code/FL_LSTM_CODE/Datasets/mobiact2.csv'
        self.BATCH_SIZE = 32
        # self.TEST_SIZE = 0.2 
        # self.VAL_SIZE = 0.2
        self.COMMUNICATION_ROUNDS = 5
        self.CLIENT_EPOCHS = 3
        self.LEARNING_RATE = 0.001
        self.HIDDEN_SIZE_BINARY = 128  
        self.HIDDEN_SIZE_MULTICLASS = 128 
        self.NUM_LAYERS = 2
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FALL_SCENARIOS = [5, 4, 10, 0]
        self.NON_FALL_SCENARIOS = [i for i in range(16) if i not in [5, 4, 10, 0]]
        self.FEDPROX_MU = 0.1
        self.FEATURE_COLUMNS = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','azimuth','pitch','roll']
        self.MAX_PARAM_SIZE = 100000
        self.SEQUENCE_LENGTH = 50
        self.SAMPLING_STRATEGY = 0.5