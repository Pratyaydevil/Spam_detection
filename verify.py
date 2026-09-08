import torch
from app.models.hybrid_model import HybridEmailDefender

def test_model():
    print("Initializing model...")
    model = HybridEmailDefender(num_meta_features=6, num_classes=3)
    
    print("Creating dummy inputs...")
    # Dummy DistilBERT inputs (batch_size=2, seq_len=10)
    input_ids = torch.randint(0, 30522, (2, 10))
    attention_mask = torch.ones((2, 10))
    
    # Dummy metadata features (batch_size=2, num_features=6)
    meta_features = torch.randn((2, 6))
    
    print("Running forward pass...")
    outputs = model(input_ids, attention_mask, meta_features)
    
    print("Outputs successfully generated!")
    for key, value in outputs.items():
        print(f" - {key}: {value.shape if hasattr(value, 'shape') else type(value)}")

if __name__ == "__main__":
    test_model()
