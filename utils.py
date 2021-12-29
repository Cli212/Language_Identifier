import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, random_split)

class InputFeature:
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask


def convert_data_to_dataloader(data, tokenizer, max_length, batch_size, seed=42):
    encoded_data = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'labels': []}
    for i, (_, texts) in enumerate(data.items()):
        encoded_batch = tokenizer.batch_encode_plus(texts, max_length=max_length, padding='max_length', truncation=True)
        encoded_data['input_ids'].extend(encoded_batch['input_ids'])
        encoded_data['attention_mask'].extend(encoded_batch['attention_mask'])
        encoded_data['token_type_ids'].extend(encoded_batch['token_type_ids'])
        # encoded_data['lengths'].extend([sum(i) for i in encoded_batch['attention_mask']])
        encoded_data['labels'].extend([i]*len(texts))
    all_input_ids = torch.tensor(encoded_data['input_ids'], dtype=torch.long)
    all_token_type_ids = torch.tensor(encoded_data['token_type_ids'], dtype=torch.long)
    all_attention_mask = torch.tensor(encoded_data['attention_mask'], dtype=torch.long)
    all_label_ids = torch.tensor(encoded_data['labels'], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    print('Train dataloader:', len(sampler), len(train_dataloader))
    print('Val dataloader:', len(val_dataloader))
    return train_dataloader, val_dataloader


def training(train_dataloader, model, device, optimizer, scheduler, max_grad_norm, epoch, writer, scaler=None, gradient_accumulation_steps=1):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(*batch)
        else:
            loss = model(*batch)
        writer.add_scalar('scalar/classification_loss', loss, epoch * len(train_dataloader) + step)
        loss /= gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # track train loss
        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # update learning rate
            scheduler.step()

            optimizer.zero_grad()
    return model, optimizer, scheduler

def evaluation(val_dataloader, model, device):
    acc_count = 0
    model.eval()
    all_preds = []
    all_labels = []
    for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluation")):
        # add batch to gpu
        labels = batch[-1]
        batch = tuple(t.to(device) for t in batch[:-1])

        with torch.no_grad():
            preds = model(*batch).detach().cpu()
            all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        acc_count += torch.sum(preds == labels).detach().cpu().tolist()
    return acc_count / len(val_dataloader.dataset), all_preds, all_labels


def inference(texts, model, tokenizer, max_length, device):
    encoded_data = tokenizer.batch_encode_plus(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    # lengths = [sum(i) for i in encoded_data['attention_mask']]
    encoded_batch = {key: value.to(device) for key, value in encoded_data.items()}
    model.eval()
    with torch.no_grad():
        # preditions = model(encoded_data['input_ids'].to(device), lengths).detach().cpu().tolist()
        preditions = model(**encoded_batch).detach().cpu().tolist()
    return preditions
