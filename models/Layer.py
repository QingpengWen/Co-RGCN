# -*- coding: utf-8 -*-#
"""
@CreateTime :       2023/2/28 23:15
@Author     :       Qingpeng Wen
@File       :       Layer.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2023/3/25 16:57
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.__args = args

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,
            self.__args.attention_hidden_dim,
            self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        return hiddens

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, args, embeddings, mem_dim, layers, dropout):
        super(GCN, self).__init__()

        self.__args = args
        self.use_cuda = self.__args.gpu
        self.mem_dim = mem_dim
        self.in_dim = embeddings
        self.in_drop = dropout
        self.dropout = dropout

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, gcn_inputs, layers):

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        for l in range(layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_outputs = gAxW
        gcn_outputs = F.dropout(gcn_outputs, self.dropout, training=self.training)

        return gcn_outputs

class CoRegression(nn.Module):
    def __init__(self, input_intent_size, input_slot_size, output_intent_size, dropout_rate, device):
        super().__init__()
        self.bilinear_SF = nn.Bilinear(in1_features=input_intent_size, in2_features=input_slot_size,
                                       out_features=1, bias=None)
        self.bilinear_ID = nn.Bilinear(in1_features=input_slot_size, in2_features=input_intent_size,
                                       out_features=1, bias=None)
        self.__dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, c_I, c_S, dropout=True):
        '''
        x = intent
        y = slot
        f_I = softmax()
        f_S = sigmoid()
        '''
        u_S = self.bilinear_SF(torch.tanh(c_I), torch.tanh(c_S))

        u_S = torch.sigmoid(u_S)
        if dropout:
            u_S = self.__dropout_layer(u_S)

        u_I = self.bilinear_ID(c_S, c_I)

        u_I = torch.nn.functional.softmax(u_I, dim=1)
        if dropout:
            u_I = self.__dropout_layer(u_I)

        v_I = u_I * c_S
        y_I = v_I + c_I

        v_S = u_S * c_I
        y_S = v_S + c_S

        return y_I, y_S

class slot_Decoder(nn.Module):
    def __init__(self, hid_size, n_slots):
        super(slot_Decoder, self).__init__()
        self.n_slots = n_slots
        self.W_slot_ans = nn.Linear(in_features=hid_size, out_features=n_slots, bias=None)

    def forward(self, slot_output, seq_lens):
        flat_slot_out = torch.cat([slot_output[i][:seq_lens[i]]
                                   for i in range(0, len(seq_lens))], dim=0)
        slot_out = self.W_slot_ans(flat_slot_out)  # [batch_size*valid, n_slots]

        return slot_out

class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens

class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor

class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x

class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context