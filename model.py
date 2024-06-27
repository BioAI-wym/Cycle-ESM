import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MyBinaryClassifier(nn.Module):
    def __init__(self, model_name, num_filters=64, filter_sizes=[3, 4, 5], dropout_rate=0):
        super(MyBinaryClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.convs = nn.ModuleList([
            nn.Conv1d(self.model.config.hidden_size, num_filters, k)
            for k in filter_sizes
        ])

        self.pool = nn.MaxPool1d(2, 2)  # Adjusted to 1D pooling

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.transpose(1, 2)  # Adjust for 1D conv (batch, channels, sequence_length)

        # Apply 1D convolution, pooling, and activation function
        x = [F.relu(self.pool(conv(x))) for conv in self.convs]

        # Global max pooling
        x = [torch.max(i, 2)[0] for i in x]

        x = self.dropout(torch.cat(x, 1))

        logits = self.fc(x).squeeze(1)

        return torch.sigmoid(logits)
