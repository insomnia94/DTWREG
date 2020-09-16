from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
    Inputs:
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - output  : Variable float (batch, max_len, hidden_size * num_dirs)
    - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
    - embedded: Variable float (batch, max_len, word_vec_size)
    """
        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1)

            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
            s2r = {s: r for r, s in enumerate(sort_ixs)}
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

            assert max(input_lengths_list) == input_labels.size(1)

            sort_ixs = input_labels.data.new(sort_ixs).long()
            recover_ixs = input_labels.data.new(recover_ixs).long()

            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)
        embedded = self.input_dropout(embedded)
        embedded = self.mlp(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[recover_ixs]

            if self.rnn_type=='lstm':
                hidden = hidden[0]
            hidden = hidden[:, recover_ixs, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)

        return output, hidden, embedded


