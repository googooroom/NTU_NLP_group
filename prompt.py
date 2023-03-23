import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the dataset

import os
import tarfile
import urllib.request

# Set the URL and file paths
extract_path = "./aclImdb"

# Load the dataset

class IMDBDataset(Dataset):
    def __init__(self, file_path):
        self.sentences = []
        self.labels = []
        with open(file_path, 'r') as f:
            for line in f:
                label, sentence = line.strip().split('\t')
                self.sentences.append(sentence)
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoded_sentence = tokenizer.encode(sentence)
        return torch.tensor(encoded_sentence), torch.tensor(label)

dataset = IMDBDataset(os.path.join(extract_path, "train"))


# Split the dataset into train and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


class PromptGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PromptGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

model = PromptGenerator(input_size=768, hidden_size=128, output_size=768)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 10
best_loss = float('inf')

for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs.view(-1))
        loss.backward()
        optimizer.step()
        
    # Compute validation loss
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs.view(-1))
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_dataset)
        
    # Save checkpoint if validation loss improves
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'path/to/best_checkpoint.pt')
    
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}')



# Tokenize input text'text': 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\'s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\'t have much of a plot.', 'label': 0}
text = "This movie was really bad"
encoded_text = tokenizer.encode(text)

# Test prompt using prompt generator
with torch.no_grad():
    inputs = torch.tensor(encoded_text).unsqueeze(0).to(device)
    prompt = model(inputs).squeeze(0).cpu().numpy()

# Decode prompt back into text
generated_prompt = tokenizer.decode(prompt)
print(generated_prompt)


###### below to be implemented
# i use this as this model as it is head on top with linear layer tied with input embeddings
from transformers import GPT2LMHeadModel

# Load pre-trained GPT-2 model
gptmodel = GPT2LMHeadModel.from_pretrained('gpt2')

