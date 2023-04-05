import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer, RobertaForSequenceClassification

import argparse
import sys

from dataset import * 
from train import *
from model import *

def main(args):
    
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True


    # For T5
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5ForConditionalGeneration.from_pretrained('t5-large')

    # Initialize the soft prompt model
    prompt_length = 20
    soft_prompt_model = SoftPromptTuning(model, prompt_length)
    
    train_val_dataset = TweetDisasterDataset(args.train_path, tokenizer, args.max_len)
    num_train, num_val = round(0.8 * len(train_val_dataset)), round(0.2 * len(train_val_dataset))
    train_dataset, val_dataset = random_split(train_val_dataset, [num_train, num_val])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epochs)
    
    if args.test:
        test_dataset = TweetDisasterDataset(args.test_path, tokenizer, args.max_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    train(soft_prompt_model, tokenizer, train_loader, val_loader, optimizer, args.epochs)

    
def parse_args(args):
    parser = argparse.ArgumentParser(description="Disaster Tweet Classification")

    
    # Required
    parser.add_argument("--train_path", type=str, help="path to training data")

    
    parser.add_argument("--instruction_path", type=str, default=None, help="path to instruction file")

    # Optional
    parser.add_argument("--test", action="store_true", default=False,
                        help="run evaluation on test data")

    parser.add_argument("--test_path", type=str, default=None,
                        help="path to test data")
    # Model hyperparameters
    parser.add_argument("--max_len", type=int, default=64,
                        help="maximum sequence length for input tokens")
    # parser.add_argument("--lr", type=float, default=2e-5,
    #                     help="learning rate for Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of epochs for training")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="weight decay for Adam optimizer")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for Adam optimizer")

        
    args = parser.parse_args()

    return args
    
    
if __name__ =="__main__":
    print(f"arguments: {sys.argv[1:]}")
    
    args = parse_args(sys.argv[1:])

    main(args)