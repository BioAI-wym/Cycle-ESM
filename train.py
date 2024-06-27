import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from sklearn.metrics import roc_auc_score

from model import MyBinaryClassifier
from dataset import DNADataset

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_file_path = 'Data/1/main/pos.txt'
neg_file_path = 'Data/1/main/neg.txt'
test_pos_file_path = 'Data/1/val/pos.txt'
test_neg_file_path = 'Data/1/val/neg.txt'

model_name = "facebook/esm2_t12_35M_UR50D"
num_epochs = 50
max_sequence_length = 100
batch_size = 32
scaler = GradScaler()
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = DNADataset(pos_file_path=pos_file_path, neg_file_path=neg_file_path, tokenizer=tokenizer, max_length=max_sequence_length)
test_dataset = DNADataset(pos_file_path=test_pos_file_path, neg_file_path=test_neg_file_path, tokenizer=tokenizer, max_length=max_sequence_length)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = MyBinaryClassifier(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=0.000065, weight_decay=0.000065)
criterion = nn.BCELoss()
accumulation_steps = 2  # Set this to the desired number of accumulation steps

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Initialize gradient accumulation
    train_loss_accumulated = 0.0  # To keep track of loss over accumulation steps

    for i, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)

        train_predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        train_loss = criterion(train_predictions, labels) / accumulation_steps  # Normalize the loss
        train_loss.backward()  # Accumulate gradients
        train_loss_accumulated += train_loss.item()

        # Perform optimization step after the specified number of accumulation steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients

    # Print average training loss per epoch
    average_train_loss = train_loss_accumulated / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Training Loss: {average_train_loss}')

    # Validation/Evaluation Step
    model.eval()
    all_valid_labels = []
    all_valid_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)

            valid_predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            all_valid_labels.extend(labels.cpu().numpy())
            all_valid_predictions.extend(valid_predictions.cpu().detach().numpy())

        valid_auc = roc_auc_score(all_valid_labels, all_valid_predictions)
        print(f'Epoch {epoch+1}/{num_epochs}, Valid AUC: {valid_auc}')
