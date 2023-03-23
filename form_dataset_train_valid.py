import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load the IMDB dataset
dataset = load_dataset('imdb')

print(type(dataset))

# Instantiate the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define the maximum sequence length
max_length = 512

# Define a custom Dataset class
class MyPromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        prompt = "What is your overall sentiment of this movie?"

        # Encode the prompt and text
        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        encoded_text = self.tokenizer.encode(text, add_special_tokens=False)

        # Truncate the inputs if they exceed the maximum sequence length
        if len(encoded_prompt) + len(encoded_text) > self.max_length:
            encoded_text = encoded_text[:self.max_length - len(encoded_prompt)]

        #input_text = torch.LongTensor(encoded_prompt + encoded_text)
        input_text = encoded_prompt + encoded_text

        # Create the attention mask
        attention_mask = torch.ones_like(input_text) #all taken 

        return {'input': input_text, 'attention_mask': attention_mask, 'label': label}

train_dataset = dataset['train']
test_dataset = dataset['test']

print(">>>>>>")

# Create the custom Dataset objects for train and validation sets
train_dataset = MyPromptDataset(train_dataset, tokenizer, max_length)
train_prompt_dataset, val_prompt_dataset = train_test_split(train_dataset, test_size=0.2)#, random_state=42)
test_prompt_dataset = MyPromptDataset(test_dataset, tokenizer, max_length)
print("<<<<<<<")
# Define the batch size and create the DataLoader objects for train and validation sets
batch_size = 8
train_loader = DataLoader(train_prompt_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_prompt_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_prompt_dataset, batch_size=batch_size)

'''
{
    'train': [
        {
            'text': "I loved this movie!",
            'label': 1
        },
        {
            'text': "This film was a complete waste of time.",
            'label': 0
        },
        ...
    ],
    'test': [
        {
            'text': "The story was well-written and the acting was superb.",
            'label': 1
        },
        {
            'text': "I really disliked this film. The characters were poorly developed and the pacing was slow.",
            'label': 0
        },
        ...
    ]
}

'''