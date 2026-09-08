import torch
import torch.optim as optim
from .loss import SCARLoss

class ModelTrainer:
    def __init__(self, model, learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = SCARLoss(num_classes=3)
        
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack batch (simulated unpacking here)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        meta_features = batch['meta_features'].to(self.device)
        labels = batch['labels'].to(self.device) # should be one-hot encoded for evidential loss
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask, meta_features)
        alpha = outputs['alpha']
        
        # Simulated adversarial text representations for SCAR Loss
        # In a real training loop, we'd run an obfuscated version of the text through text_pipeline
        # to get adv_feats, and compare it to text_feats. Here we simulate it.
        # This assumes we have anchor and adv features available if we ran them separately
        
        # Loss calculation
        loss, loss_ev, loss_sc = self.criterion(alpha, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_ev.item(), loss_sc.item()
        
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                meta_features = batch['meta_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, meta_features)
                loss, _, _ = self.criterion(outputs['alpha'], labels)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
