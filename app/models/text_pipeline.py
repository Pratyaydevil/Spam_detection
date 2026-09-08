import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class TextPipeline(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", lstm_hidden=128, dropout=0.3):
        super(TextPipeline, self).__init__()
        
        # We use a transformer for semantic understanding
        self.transformer = DistilBertModel.from_pretrained(model_name)
        self.transformer_config = DistilBertConfig.from_pretrained(model_name)
        
        # BiLSTM for sequential and urgency patterns
        self.bilstm = nn.LSTM(
            input_size=self.transformer_config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Final projection to a fixed size embedding
        self.fc = nn.Linear(lstm_hidden * 2, 256)
        
    def forward(self, input_ids, attention_mask):
        # 1. DistilBERT extraction
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Sequence of hidden states: (batch_size, seq_length, hidden_size)
        sequence_output = outputs.last_hidden_state
        
        # 2. BiLSTM mapping
        lstm_out, _ = self.bilstm(sequence_output)
        
        # 3. Pooling (taking the last hidden state of the LSTM)
        # Using max pooling over the sequence length
        pooled_out, _ = torch.max(lstm_out, dim=1)
        
        # 4. Dropout and Projection
        out = self.dropout(pooled_out)
        out = self.fc(out)
        
        return out
