import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class SoftPromptTuning(nn.Module):
    def __init__(self, model: T5ForConditionalGeneration, prompt_length: int):
        super(SoftPromptTuning, self).__init__()
        self.model = model
        self.prompt_length = prompt_length
        self.soft_prompt_embeddings = nn.Parameter(torch.randn(prompt_length, self.model.config.d_model))

        # Freeze the T5 model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Prepend the soft prompt embeddings to the input embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        prompt_embeddings = self.soft_prompt_embeddings.unsqueeze(0).repeat(input_embeddings.size(0), 1, 1)
        combined_embeddings = torch.cat((prompt_embeddings, input_embeddings), dim=1)

        # Adjust the attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.cat((torch.ones(input_embeddings.size(0), self.prompt_length).to(attention_mask.device), attention_mask), dim=1)
        # Feed the combined embeddings through the model
        outputs = self.model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels)

        return outputs
