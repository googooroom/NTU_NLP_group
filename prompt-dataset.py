import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
batch_size = 16

train_texts = [
    "I love this movie. It's so funny and heartwarming.",
    "This is the worst movie I've ever seen. The acting is terrible and the plot makes no sense.",
    "The special effects in this movie are amazing. It's a visual feast for the eyes.",
    "I found this movie to be incredibly boring. Nothing happens for the first hour.",
    "The soundtrack in this movie is fantastic. It really adds to the overall experience.",
    "The dialogue in this movie is really well-written. It feels natural and flows smoothly."
]

prompts = [
    "What did you think of the movie?",
    "How would you rate the acting?",
    "What did you think of the special effects?",
    "Did you find the movie entertaining?",
    "What did you think of the soundtrack?",
    "How was the dialogue in the movie?"
]



class PromptDataset(Dataset):
    def __init__(self, input_texts, prompts):
        self.input_texts = input_texts
        self.prompts = prompts

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        input_text = self.input_texts[idx]

        # Concatenate prompt and input text
        input_sequence = prompt + input_text

        # Convert inputs to tensors
        input_ids = torch.tensor(tokenizer.encode(input_sequence))
        label = torch.tensor(1)  # Placeholder label (not used in training)

        return {'input_ids': input_ids, 'labels': label}


#Trainer to be used in this case
from transformers import TrainingArguments, Trainer


def data_collator(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad inputs to same length
    input_ids = pad_sequence(input_ids, batch_first=True)

    return {'input_ids': input_ids, 'labels': torch.stack(labels)}



train_dataset = PromptDataset(input_texts=train_texts, prompts=prompts)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

print("done")
