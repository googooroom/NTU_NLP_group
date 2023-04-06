import torch
import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

import torch
from sklearn.metrics import accuracy_score


def validate_epoch(model, dataloader, tokenizer, max_len, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            text_input = batch["text_input"]
            label = batch["label"]
            
            # Tokenize the text_input
            encoding = tokenizer(text_input, add_special_tokens=True, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Convert labels to tensor and move to device
            labels = label.clone().detach().to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate predictions
            predicted_tokens = outputs.logits.argmax(dim=-1)

            # Calculate the accuracy
            batch_size = len(labels)
            accuracy = (predicted_tokens == labels).sum().item() / batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / total_samples
    
        
    gc.collect()
    torch.cuda.empty_cache()

    return average_loss, average_accuracy


def train_epoch(model, dataloader, optimizer, scheduler, tokenizer, max_len, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    for batch in dataloader:
        text_input = batch["text_input"]
        label = batch["label"]
        
        # Tokenize the text_input
        encoding = tokenizer(text_input, add_special_tokens=True, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Convert labels to tensor and move to device
        labels = label.clone().detach().to(device)
                
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Generate predictions
        predicted_tokens = outputs.logits.argmax(dim=-1)

        # Calculate the accuracy
        batch_size = len(labels)
        accuracy = (predicted_tokens == labels).sum().item() / batch_size
        total_accuracy += accuracy * batch_size
        total_samples += batch_size
        
    scheduler.step()
    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / total_samples
        
    gc.collect()
    torch.cuda.empty_cache()

    return average_loss, average_accuracy



def train(soft_prompt_model, train_dataloader, val_dataloader, optimizer, scheduler, tokenizer, max_len, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    soft_prompt_model.to(device)
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(soft_prompt_model, train_dataloader, optimizer, scheduler, tokenizer, max_len, device)
        val_loss, val_accuracy = validate_epoch(soft_prompt_model, val_dataloader, tokenizer, max_len, device)
        print(f'Epoch {epoch + 1}/{epochs} -- LR: {scheduler.get_last_lr()[0]:.6f} -- Training Loss: {train_loss:.4f} -- Training Accuracy: {train_accuracy:.4f} -- Validation Loss: {val_loss:.4f} -- Validation Accuracy: {val_accuracy:.4f}')
    
    
    gc.collect()
    torch.cuda.empty_cache()

    
    
    
def test_epoch(model, dataloader, tokenizer,device):
    model.eval()
    total_accuracy = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            # Generate predictions
            predicted_tokens = outputs.logits.argmax(dim=-1)

            # Calculate the accuracy
            batch_size = len(labels)
            accuracy = (predicted_tokens == labels).sum().item() / batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size

    average_accuracy = total_accuracy / total_samples
    return average_accuracy