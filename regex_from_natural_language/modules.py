# Modified from https://github.com/keon/seq2seq

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, src):
        """
        Args:
            src: (src_len, batch_size)

        Returns:
            outputs: (src_len, batch_size, hidden_dim)
            hidden: (num_layers * 2, batch_size, hidden_dim)
        """
        src = src.transpose(0, 1)  # (batch_size, src_len)
        embedded = self.embedding(src)  # (batch_size, src_len, embedding_dim)
        embedded = self.dropout(embedded)  # (batch_size, src_len, embedding_dim)
        embedded = embedded.transpose(0, 1)  # (src_len, batch_size, embedding_dim)

        outputs, hidden = self.rnn(
            embedded
        )  # (src_len, batch_size, hidden_dim * 2), (num_layers * 2, batch_size, hidden_dim)
        outputs = (
            outputs[:, :, : self.hidden_dim] + outputs[:, :, self.hidden_dim :]
        )  # (src_len, batch_size, hidden_dim)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        std_v = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-std_v, std_v)

    def forward(self, hidden, encoder_outputs, src_mask):
        """
        Args:
            hidden: (1, batch_size, hidden_dim)
            encoder_outputs: (src_len, batch_size, hidden_dim)
            src_mask: (src_len, batch_size)

        Returns:
            (batch_size, 1, src_len)
        """
        src_len = encoder_outputs.size(0)
        h = hidden.repeat(src_len, 1, 1).transpose(0, 1)  # (batch_size, src_len, hidden_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch_size, src_len, hidden_dim)
        attn_energies = self.score(h, encoder_outputs)  # (batch_size, src_len)
        if src_mask is not None:
            attn_energies = attn_energies.masked_fill(~src_mask.transpose(0, 1), float("-inf"))
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # (batch_size, 1, src_len)

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # (batch_size, src_len, hidden_dim)
        energy = energy.transpose(1, 2)  # (batch_size, hidden_dim, src_len)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        energy = torch.bmm(v, energy)  # (batch_size, 1, src_len)
        return energy.squeeze(1)  # (batch_size, src_len)


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embedding_dim, hidden_dim, num_layeres, dropout):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layeres = num_layeres

        self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim, num_layeres, dropout=dropout)
        self.fc = nn.Linear(embedding_dim + hidden_dim * 2, tgt_vocab_size)

    def forward(self, input, prev_hidden, encoder_outputs, src_mask):
        """
        Args:
            input: (batch_size)
            prev_hidden: (num_layers, batch_size, hidden_dim)
            encoder_outputs: (src_len, batch_size, hidden_dim)
            src_mask: (src_len, batch_size)

        Returns:
            output: (batch_size, hidden_dim)
            hidden: (num_layers, batch_size, hidden_dim)
            attn_weights: (batch_size, 1, src_len)
        """
        embedded = self.embedding(input).unsqueeze(0)  # (1, batch_size, embedding_dim)
        embedded = self.dropout(embedded)  # (1, batch_size, embedding_dim)

        attn_weights = self.attention(prev_hidden[-1], encoder_outputs, src_mask)  # (batch_size, 1, src_len)
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))  # (batch_size, 1, hidden_dim)
        context = context.transpose(0, 1)  # (1, batch_size, hidden_dim)

        rnn_input = torch.cat([embedded, context], 2)  # (1, batch_size, embedding_dim + hidden_dim)
        output, hidden = self.rnn(
            rnn_input, prev_hidden
        )  # (1, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim)

        embedded = embedded.squeeze(0)  # (batch_size, embedding_dim)
        output = output.squeeze(0)  # (batch_size, hidden_dim)
        context = context.squeeze(0)  # (batch_size, hidden_dim)
        output = self.fc(torch.cat([embedded, output, context], 1))  # (batch_size, tgt_vocab_size)
        return output, hidden, attn_weights
