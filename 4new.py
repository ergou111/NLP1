import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
all_letters = all_letters + all_letters.lower() + "_"
n_letters = len(all_letters)

def letter_to_index(letter: str) -> int:
    idx = all_letters.find(letter)
    if idx == -1:
        return all_letters.find("_")
    return idx

def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), n_letters, dtype=torch.float32)
    for i, letter in enumerate(line):
        idx = letter_to_index(letter)
        tensor[i][idx] = 1.0
    return tensor

def read_lines(filename: str):
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [line.strip() for line in lines if line.strip()]

class NamesDataset(Dataset):
    def __init__(self, data_folder: str):
        self.data = []
        self.category_names = []

        files = glob.glob(os.path.join(data_folder, '*.txt'))
        files.sort()

        for filename in files:
            category = os.path.splitext(os.path.basename(filename))[0]
            self.category_names.append(category)

        self.num_categories = len(self.category_names)
        self.category_to_index = {
            category: i for i, category in enumerate(self.category_names)
        }

        for filename in files:
            category = os.path.splitext(os.path.basename(filename))[0]
            category_index = self.category_to_index[category]
            names = read_lines(filename)

            for name in names:
                self.data.append((name, category_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, category_index = self.data[idx]
        input_tensor = line_to_tensor(name).unsqueeze(1)
        target_tensor = torch.tensor([category_index], dtype=torch.long)
        return input_tensor, target_tensor

    def get_category_name(self, category_index: int) -> str:
        return self.category_names[category_index]

    def get_raw_item(self, idx):
        name, category_index = self.data[idx]
        return name, self.category_names[category_index]

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # line_tensor: [seq_len, 1, input_size]
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output

def make_batches(n_samples, batch_size):
    indices = list(range(n_samples))
    random.shuffle(indices)
    return [indices[i:i + batch_size] for i in range(0, n_samples, batch_size)]

def train_one_epoch(model, dataset, optimizer, loss_fn, batch_size=32):
    model.train()
    batches = make_batches(len(dataset), batch_size)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in batches:
        optimizer.zero_grad()
        batch_loss = 0

        for i in batch:
            input_tensor, target_tensor = dataset[i]
            output = model(input_tensor)
            loss = loss_fn(output, target_tensor)
            batch_loss += loss

            pred = output.argmax(dim=1).item()
            real = target_tensor.item()
            if pred == real:
                correct += 1
            total += 1

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    avg_loss = total_loss / len(batches)
    accuracy = correct / total
    return avg_loss, accuracy

def predict(model, dataset, name):
    model.eval()
    with torch.no_grad():
        line_tensor = line_to_tensor(name).unsqueeze(1)
        output = model(line_tensor)
        pred_index = output.argmax(dim=1).item()
        return dataset.get_category_name(pred_index)

if __name__ == '__main__':
    data_folder = 'names'
    dataset = NamesDataset(data_folder)

    print('Number of samples:', len(dataset))
    print('Number of categories:', dataset.num_categories)
    print('Categories:', dataset.category_names)

    input_tensor, target_tensor = dataset[0]
    raw_name, raw_category = dataset.get_raw_item(0)

    print('\nFirst sample:')
    print('Name:', raw_name)
    print('Country:', raw_category)
    print('Input tensor shape:', input_tensor.shape)
    print('Target tensor:', target_tensor)

    n_hidden = 128
    model = CharRNN(n_letters, n_hidden, dataset.num_categories)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 10
    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, dataset, optimizer, loss_fn, batch_size=32)
        print(f'Epoch {epoch:2d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}')

    print('\nPrediction examples:')
    for i in range(min(10, len(dataset))):
        name, real_category = dataset.get_raw_item(i)
        pred_category = predict(model, dataset, name)
        print(f'Name: {name:15s} | Real: {real_category:10s} | Pred: {pred_category}')