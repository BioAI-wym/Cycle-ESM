import torch
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, pos_file_path, neg_file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load positive and negative sequences
        with open(pos_file_path, 'r') as file:
            pos_sequences = file.readlines()

        with open(neg_file_path, 'r') as file:
            neg_sequences = file.readlines()

        # Combine sequences and labels
        self.sequences = pos_sequences + neg_sequences
        self.labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }
