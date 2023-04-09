import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, T5Tokenizer, T5ForConditionalGeneration, \
    RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, \
    GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification

import argparse
import sys
import time

from dataset import * 
from train import *
from model import *
from utils import *

def main(args):

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = 2
    print('--Load Models')
    # Create the model instance
    pretrain_name = 'roberta-large'
    config = RobertaConfig.from_pretrained(pretrain_name, num_labels=labels)
    model = RobertaForSequenceClassification.from_pretrained(pretrain_name, config=config)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_name)

    # config.num_labels = 2

    # Load the pre-trained model and tokenizer    
    # model = RobertaForSequenceClassification.from_pretrained("roberta-large", config=config)
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    # Initialize the soft prompt model
    if args.apply_soft_prompt:        
        model = SoftPromptTuning(model, args.prompt_length)

    
    print('--Load data')
    
    train_val_dataset = TweetDisasterDataset(args.train_path)
    num_train, num_val = round(0.8 * len(train_val_dataset)), round(0.2 * len(train_val_dataset))
    train_dataset, val_dataset = random_split(train_val_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    if args.baseline:
        print('--Running Baseline')
        evaluate_base(model, val_loader, args.max_len, tokenizer)
    else:
        print('--Begin Training')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epochs)

        start_time = time.time()
        train_result = train(model, train_loader, val_loader, optimizer, scheduler, tokenizer, args.max_len, args.epochs)
        time_taken = start_time - time.time()
        print(f"Training Time: {time_taken:.2f}")

        plotter(train_result, args.fig_name)

        np.savez(os.path.join('./diagram', args.fig_name.replace('.png ', '.npz')), train_loss=train_result['train_loss'], val_loss=train_result['val_loss'], train_acc=train_result['train_acc'], val_acc=train_result['val_acc'])

    if args.test:
        test_dataset = TweetDisasterDataset(args.test_path)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    
        test_accuracy = test_epoch(model, test_loader, tokenizer, device)
        print(f"Test accuracy: {test_accuracy:.4f}")

    
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
    parser.add_argument('--fig_name',type=str, help='')

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
    parser.add_argument("--prompt_length", type=int, default=20,
                    help="Length of the soft prompt") 
    
    # apply --apply_soft_prompt will become true in cmd line
    parser.add_argument("--apply_soft_prompt", default=False, action='store_true',
                    help="use soft prompt tuning")   
    parser.add_argument("--baseline", default=False, action='store_true',
                    help="use soft prompt tuning")   
    
    parser.add_argument("--net", type=str, default='roberta',
                    help="set the pretrained model [roberta, gpt2, bert-base-uncased]") 
    
    args = parser.parse_args()

    return args
    
    
if __name__ =="__main__":
    print(f"arguments: {sys.argv[1:]}")
    
    args = parse_args(sys.argv[1:])

    main(args)