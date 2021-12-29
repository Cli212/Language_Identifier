import torch
from torch import nn
from transformers import AutoModel


class Identifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_size, num_labels, num_layers=1, bidirectional=False, batch_first=True, dropout=0.1):
        super(Identifier, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # self.rnn = nn.RNN(emb_dim, hid_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.model = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Linear(hid_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # lengths = torch.sum(attention_mask, dim=-1)
        # embeddings = self.embedding(input_ids)
        # embed_input_x_packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        # encoder_outputs_packed, h_last = self.rnn(embed_input_x_packed)
        # encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        # h_last = torch.sum(h_last, dim=0)
        # logits = self.classifier(h_last)
        outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        logits = self.classifier(outputs[:, 0])
        if labels is not None:
            ## training
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            loss = criterion(self.softmax(logits), labels)
            return loss
        else:
            ## evaluation
            return torch.argmax(logits, dim=-1)
