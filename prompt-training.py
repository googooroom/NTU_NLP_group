import torch
import math
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from torch.optim import Adam


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': torch.tensor(label)}
        
train_texts = ["I loved the movie!", "The movie was terrible.", "The acting was amazing."]
train_labels = [1, 0, 1]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
max_len = 32

train_dataset = MyDataset(train_texts, train_labels, tokenizer, max_len)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = None
optimizer = Adam(model.parameters(), lr=0.001)

log_interval = 10   # every 10 run once validation
batch_idx = 0
for batch in train_dataloader:
    batch_idx += 1
    train_loss = 0

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']
    
    # training model here

    # Reset gradients
    optimizer.zero_grad()

    # Forward pass  -- GPT-2 model
    loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)

    # Backward pass
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    # Calculate training perplexity
    train_ppl = math.exp(train_loss / (batch_idx + 1))

    # Print training loss and perplexity
    if batch_idx % log_interval == 0:
        print(f'Train batch {batch_idx}: loss = {train_loss:.4f}, perplexity = {train_ppl:.2f}')



