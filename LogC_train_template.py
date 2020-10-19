import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 48
num_epochs = 300
batch_size = 256
model_dir = '/'
log = 'Log_Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(name) as f:
        for line in f.readlines():
            line = tuple(map(int, line.strip().split()))
            if len(line) > window_size:
                num_sessions += 1
                for i in range(len(line) - window_size):
                    inputs.append(line[i:i + window_size])
                    outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
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

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate('normal_logkey_train.txt')
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
    torch.save(model.state_dict(), model_dir + log)
    print('Finished Training')
