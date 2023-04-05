import torch
import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

import torch
from sklearn.metrics import accuracy_score




# Validation function
def validate_epoch(model,tokenizer, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            total_loss += loss.item()
        # Generate predictions
            predicted_tokens = outputs.logits.argmax(dim=-1)

            # Decode the predicted tokens and labels
            predicted_labels = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            true_labels = [tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels]


            # Calculate the accuracy
            accuracy = sum([1 if pred == true else 0 for pred, true in zip(predicted_labels, true_labels)]) / len(predicted_labels)
            total_accuracy += accuracy

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)
    return average_loss, average_accuracy



def train_epoch(model, tokenizer, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        
        optimizer.zero_grad()
        outputs = model.forward(input_ids, attention_mask, labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

          # Generate predictions
        predicted_tokens = outputs.logits.argmax(dim=-1)
        
        # Decode the predicted tokens and labels
        predicted_labels = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
        true_labels = [tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels]
        
        # Calculate the accuracy
        accuracy = sum([1 if pred == true else 0 for pred, true in zip(predicted_labels, true_labels)]) / len(predicted_labels)
        total_accuracy += accuracy

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)
    return average_loss, average_accuracy

def train(soft_prompt_model, tokenizer,train_dataloader, val_dataloader, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    soft_prompt_model.to(device)
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(soft_prompt_model, tokenizer, train_dataloader, optimizer, device)
        val_loss, val_accuracy = validate_epoch(soft_prompt_model, tokenizer, val_dataloader, device)
        print(f'Epoch {epoch + 1}/{epochs} -- Training Loss: {train_loss:.4f} -- Training Accuracy: {train_accuracy:.4f} -- Validation Loss: {val_loss:.4f} -- Validation Accuracy: {val_accuracy:.4f}')
    
    
    gc.collect()
    torch.cuda.empty_cache()
# def evalute(tesr, model, optimizer, scheduler, epochs):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     for epoch in range(epochs):
#         train_loss = 0
#         train_acc = 0
#         train_examples = 0 
        
#         val_examples = 0 
#         val_loss = 0
#         val_acc = 0 
        
#         model.train()
      

#         for step, batch in enumerate(train_loader):
#             input_ids = batch["input_ids"].to(device)
#             labels = batch["label"].to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)

#             train_acc += torch.sum(preds == labels).item()

#             train_loss += loss.item() * input_ids.size(0)
#             train_examples += input_ids.size(0)

#         train_loss /= train_examples
#         train_acc /= train_examples

#         model.eval()
#         for batch in val_loader:
#             input_ids = batch["input_ids"].to(device)
#             labels = batch["label"].to(device)
#             batch_size = input_ids.size(0)

#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs.loss

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)

#             val_acc += torch.sum(preds == labels).item()

#             val_loss += loss.item() * batch_size
#             val_examples += batch_size
            
#         val_loss /= val_examples
#         val_acc /= val_examples

#         print(f"Epoch: {(epoch+1):d}/{epochs:d}.. Learning Rate: {scheduler.get_last_lr()[0]:.7f}.. Train Loss : {train_loss:.4f}.. Train Acc: {train_acc:.4f}.. Val Loss : {val_loss:.4f}.. Val Acc: {val_acc:.4f}")

#         gc.collect()
#         torch.cuda.empty_cache()
#         scheduler.step()




