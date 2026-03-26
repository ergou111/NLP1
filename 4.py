import os
import glob
import torch
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
        if idx != -1:
            tensor[i][idx] = 1.0
    return tensor


def category_to_onehot(category_index: int, num_categories: int) -> torch.Tensor:
    tensor = torch.zeros(num_categories, dtype=torch.float32)
    tensor[category_index] = 1.0
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

        if len(files) == 0:
            raise ValueError(f'No .txt files found in folder: {data_folder}')

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
                if len(name) > 0:
                    self.data.append((name, category_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, category_index = self.data[idx]

        input_tensor = line_to_tensor(name)
        target_tensor = category_to_onehot(category_index, self.num_categories)

        return input_tensor, target_tensor

    def get_category_name(self, category_index: int) -> str:
        return self.category_names[category_index]

    def get_raw_item(self, idx):
        name, category_index = self.data[idx]
        return name, self.category_names[category_index]


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
    print('Input one-hot:', input_tensor)
    print('Target tensor shape:', target_tensor.shape)
    print('Target one-hot:', target_tensor)


    print('\nSome examples:')
    for i in range(min(5, len(dataset))):
        name, category = dataset.get_raw_item(i)
        x, y = dataset[i]
        print(f'{i}: {name:15s} -> {category:10s} | x shape={tuple(x.shape)}, y={y.tolist()}')