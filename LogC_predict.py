import torch
import torch.nn as nn

import time
import argparse
from tqdm import tqdm
import os

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda")
# Hyperparameters
window_size = 10
# LogKeyModel
key_input_size = 1
key_num_layers = 2
key_hidden_size = 64
key_top_k = 9
key_class = 48
key_model_path = ''

# ComponentModel
cpn_input_size = 1
cpn_num_layers = 1
cpn_hidden_size = 64
cpn_top_k = 6
cpn_class = 9
cpn_model_path = ''


def generate(path):   
    hdfs = []
    with open(path, 'r') as f:
        for ln in f.readlines():
            ln = list(map(int, ln.strip().split()))
            if len(ln) > window_size:
                hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(path, len(hdfs)))
    return hdfs

class LogKeyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(LogKeyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ComponentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(ComponentModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':

    # LogkeyModel & ComponentModel
    key_model = LogKeyModel(key_input_size, key_hidden_size, key_num_layers, key_class).to(device)
    key_model.load_state_dict(torch.load(key_model_path))
    key_model.eval()
    print('key_model_path: {}'.format(key_model_path))

    cpn_model = ComponentModel(cpn_input_size, cpn_hidden_size, cpn_num_layers, cpn_class).to(device)
    cpn_model.load_state_dict(torch.load(cpn_model_path))
    cpn_model.eval()
    print('cpn_model_path: {}'.format(cpn_model_path))

    # load log key data and component data
    key_normal_test = generate('')
    key_abnormal_test = generate('')

    cpn_normal_test = generate('')
    cpn_abnormal_test = generate('')
    FP, FN, TP, TN = 0, 0, 0, 0
    loss = []
    # Test the two models
    start_time = time.time()
    with torch.no_grad():
        for key, cpn in tqdm(zip(key_normal_test, cpn_normal_test), desc="normal test", total = len(key_normal_test)):
            for i in range(len(key) - window_size):
                # log key prediction
                key_seq = key[i:i + window_size]
                key_label = key[i + window_size]
                key_seq = torch.tensor(key_seq, dtype=torch.float).view(-1, window_size, key_input_size).to(device)
                key_label = torch.tensor(key_label).view(-1).to(device)
                key_output = key_model(key_seq)
                key_predicted = torch.argsort(key_output, 1)[0][-key_top_k:]

                # component prediction
                cpn_seq = cpn[i:i + window_size]
                cpn_label = cpn[i + window_size]
                cpn_seq = torch.tensor(cpn_seq, dtype=torch.float).view(-1, window_size, cpn_input_size).to(device)
                cpn_label = torch.tensor(cpn_label).view(-1).to(device)
                cpn_output = cpn_model(cpn_seq)
                cpn_predicted = torch.argsort(cpn_output, 1)[0][-cpn_top_k:]

                # combine the result
                if cpn_label not in cpn_predicted:
                    FP += 1
                    break
                elif key_label not in key_predicted:
                    FP += 1
                    break
    TN = len(key_normal_test) - FP

    with torch.no_grad():
        for key, cpn in tqdm(zip(key_abnormal_test, cpn_abnormal_test), desc="abnormal test", total = len(key_abnormal_test)):
            for i in range(len(key) - window_size):
                # log key prediction
                key_seq = key[i:i + window_size]
                key_label = key[i + window_size]
                key_seq = torch.tensor(key_seq, dtype=torch.float).view(-1, window_size, key_input_size).to(device)
                key_label = torch.tensor(key_label).view(-1).to(device)
                key_output = key_model(key_seq)
                key_predicted = torch.argsort(key_output, 1)[0][-key_top_k:]

                # component prediction
                cpn_seq = cpn[i:i + window_size]
                cpn_label = cpn[i + window_size]
                cpn_seq = torch.tensor(cpn_seq, dtype=torch.float).view(-1, window_size, cpn_input_size).to(device)
                cpn_label = torch.tensor(cpn_label).view(-1).to(device)
                cpn_output = cpn_model(cpn_seq)
                cpn_predicted = torch.argsort(cpn_output, 1)[0][-cpn_top_k:]

                # combine the result
                if cpn_label not in cpn_predicted:
                    TP += 1
                    break
                elif key_label not in key_predicted:
                    TP += 1
                    break
    FN = len(key_abnormal_test) - TP
    print("FP = {},FN = {},TP = {}, TN = {}".format(FP, FN, TP, TN))

    # Compute precision, recall and F1-measure
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
