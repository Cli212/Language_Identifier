import os
import json
import pandas as pd
import torch
import argparse
import itertools
import numpy as np
import datetime as dt
from tqdm import tqdm
from model import Identifier
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from transformers import (AdamW, AutoTokenizer, get_linear_schedule_with_warmup)
from utils import convert_data_to_dataloader, training, evaluation, inference
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--hid_size', default=256, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--data_path', default='./data/data_50.json')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--infer', default=False, action='store_true')
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--card_number', default=0, type=int, help='Your GPU card number.')
    parser.add_argument('--use_cpu', default=False, action="store_true")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup_proportion', default=0.0)
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--max_grad_norm', default=10)
    parser.add_argument('--infer_file_path', default=None, help="The csv file path of the file to be evaluated, the file should contain on column with column name `text`")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda:{}".format(args.card_number) if torch.cuda.is_available() and args.use_cpu is False else "cpu")
    with open(args.data_path) as f:
        data = json.load(f)
    id2code = dict(zip(range(len(data)), data.keys()))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # tokenizer = Tokenizer(BPE())
    # trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # flatten_data = list(itertools.chain(*data.values()))
    # tokenizer.train_from_iterator(flatten_data, trainer=trainer, length=len(flatten_data))
    MODEL_PATH = os.path.join(args.output_dir, 'model.pt')

    if args.train:
        train_dataloader, val_dataloader = convert_data_to_dataloader(data, tokenizer, args.max_length, args.batch_size, seed=args.seed)
        model = Identifier(tokenizer.vocab_size, args.emb_dim, args.hid_size, len(data), args.num_layers, args.bidirectional,
                           dropout=args.dropout)
        model.to(device)
        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        print(f"Number of training optimization steps: {num_train_optimization_steps}, warmup steps {warmup_steps}")

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        writer = SummaryWriter(log_dir='./log/')
        scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        for epoch in range(args.epochs):
            print("Epoch: %4i" % epoch, dt.datetime.now())

            # TRAINING
            model, optimizer, scheduler = training(train_dataloader, model=model, device=device,
                                                            optimizer=optimizer, scheduler=scheduler,
                                                            max_grad_norm=args.max_grad_norm, epoch=epoch,
                                                            writer=writer,
                                                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                            scaler=scaler)
            acc, _, _ = evaluation(val_dataloader, model, device)
            print("Accuracy is %.2f" % acc)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(model.state_dict(), MODEL_PATH)
    if args.eval and not args.train:
        _, val_dataloader = convert_data_to_dataloader(data, tokenizer, args.max_length, args.batch_size,
                                                                      seed=args.seed)
        # Model
        model = Identifier(tokenizer.vocab_size, args.emb_dim, args.hid_size, len(data), args.num_layers,
                           args.bidirectional,
                           dropout=args.dropout)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.to(device)
        acc, preds, labels = evaluation(val_dataloader, model, device)
        flatten_data = list(itertools.chain(*data.values()))
        texts = [flatten_data[i] for i in val_dataloader.dataset.indices]
        with open('./data/languages.json') as f:
            code2lan = json.load(f)
        preds = [code2lan[id2code[i]] for i in preds]
        labels = [code2lan[id2code[i]] for i in labels]
        print("Accuracy is %.2f" % acc)
        pd.DataFrame({'text': texts, 'target_label': labels, 'prediction': preds}).to_excel('evaluation_result_with_acc_%.2f.xlsx' % acc)

    # Load the fine-tuned model:
    if args.infer:
        # Model
        model = Identifier(tokenizer.vocab_size, args.emb_dim, args.hid_size, len(data), args.num_layers,
                           args.bidirectional,
                           dropout=args.dropout)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.to(device)
        with open('./data/languages.json') as f:
            code2lan = json.load(f)
        if args.infer_file_path is not None:
            df = pd.read_csv(args.infer_file_path)
            texts = df['text'].values.tolist()
            result = inference(texts, model, tokenizer, args.max_length, device)
            result = [code2lan[id2code[i]] for i in result]
            df['preds'] = result
            df.to_csv('preds.csv')
        else:
            while True:
                input_text = input("Please input your sentence:\n")
                if input_text not in ['q', 'quit', '']:
                    result = inference([input_text], model, tokenizer, args.max_length, device)[0]
                    print(f'Your input sentence seems to be in {code2lan[id2code[result]]}')
                else:
                    break


if __name__ == '__main__':
    args = parser()
    main(args)

