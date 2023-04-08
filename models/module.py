# -*- coding: utf-8 -*-#
"""
@CreateTime :       2023/2/28 22:16
@Author     :       Qingpeng Wen
@File       :       module.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2023/3/10 19:25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dependency import dependency_tree
from models.Layer import Encoder, GCN, LSTMEncoder, CoRegression, slot_Decoder

class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args
        self.__cuda = args.gpu

        # Initialize an embedding object.
        self.__embedding = nn.Embedding(
            self.__num_word,
            self.__args.word_embedding_dim
        )
        self.G_encoder = Encoder(args)

        # TODO: Slot and intent Encoder.
        self.__slot_bilstm = LSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            # self.__args.slot_decoder_hidden_dim,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.dropout_rate
        )
        self.__intent_bilstm = LSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

        self.temper = (self.__args.encoder_hidden_dim + self.__args.attention_output_dim) ** 0.5

        # Initialize an Decoder object for intent.
        self.__intent_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                      self.__args.encoder_hidden_dim + self.__args.attention_output_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim, self.__num_intent),
        )

        self.softmax = nn.Softmax(dim=-1)

        self.__intent_embedding = nn.Parameter(
            torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim))  # 191, 32
        nn.init.normal_(self.__intent_embedding.data)

        self.__CoRe = CoRegression(self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                                   self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                                   self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                                   self.__args.dropout_rate, self.__cuda)

        self.__Syngcn = GCN(self.__args,
                            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                            2, self.__args.dropout_rate)

        self.__Semgcn = GCN(self.__args,
                            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                            2, self.__args.dropout_rate)

        self.__slot_decoder = slot_Decoder(self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                                           self.__num_slot)

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def get_att(self, matrix_1, matrix_2, adjacency_matrix):
        u = torch.matmul(matrix_1.float(), matrix_2.permute(0, 2, 1).float()) / self.temper
        attention_scores = self.softmax(u)
        delta_exp_u = torch.mul(attention_scores, adjacency_matrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10).type_as(matrix_1)

        if torch.cuda.is_available():
            attention = attention.cuda()
        return attention

    def slot_adj(self, lens, valid, device):
        adjs = []
        batch = len(lens)
        allen = len(lens[-1])
        for j in range(batch):
            adj = torch.eye(allen, device=device)
            for i in range(allen):
                if i == 0:
                    adj[i, i] = 1
                    adj[i, i + 1] = 1
                elif i == valid[j]-1:
                    adj[i, i - 1] = 1
                elif i >= valid[j]:
                    adj[i, i] = 0
                else:
                    adj[i, i] = 1
                    adj[i, i - 1] = 1
                    adj[i, i + 1] = 1
            adjs.append(torch.unsqueeze(adj, dim=0))
        adj = torch.cat(adjs, dim=0)
        if torch.cuda.is_available():
            adj = adj.cuda()
        return adj

    def forward(self, text, raw_text, seq_lens, n_predicts=None):

        # TODO: Word embedding
        word_tensor = self.__embedding(text)
        g_hiddens = self.G_encoder(word_tensor, seq_lens)

        # TODO: Syc_adj_matrix
        adj_dependency = dependency_tree(raw_text, text)
        adj_output = self.get_att(word_tensor, word_tensor, adj_dependency)

        # TODO: Slot SynGCN Layer
        slot_lstm_out = self.__slot_bilstm(g_hiddens, seq_lens)
        slot_gcn_output = self.__Syngcn(adj_output, slot_lstm_out, 2)

        # TODO: Intent SemGCN Layer
        intent_adj = self.slot_adj(text, seq_lens, device=slot_gcn_output.device)
        intent_lstm_out = self.__intent_bilstm(g_hiddens, seq_lens)
        intent_gcn_output = self.__Semgcn(intent_adj, intent_lstm_out, 2)

        # TODO: Co-Regression
        logits_intent, logits_slot = self.__CoRe(intent_gcn_output, slot_gcn_output)

        # TODO: Decoder
        pred_intent = self.__intent_decoder(logits_intent)
        pred_slot = self.__slot_decoder(logits_slot, seq_lens)  # [batch*seq_lens, hidden]

        seq_lens_tensor = torch.tensor(seq_lens)
        if self.__args.gpu:
            seq_lens_tensor = seq_lens_tensor.cuda()

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)

            intent_index_sum = torch.cat(
                [
                    torch.sum(torch.sigmoid(pred_intent[i, 0:seq_lens[i], :]) > self.__args.threshold, dim=0).unsqueeze(
                        0)
                    for i in range(len(seq_lens))
                ],
                dim=0
            )
            intent_index = (intent_index_sum > torch.div(seq_lens_tensor, 2, rounding_mode='trunc').unsqueeze(1)).nonzero()

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()

