# -*- coding: utf-8 -*-#
"""
@CreateTime :       2023/2/28 20:28
@Author     :       Qingpeng Wen
@File       :       dependency.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2023/2/28 23:25
"""

import torch
import spacy
import numpy as np

# Please use the version that matches 'spacy.version'
nlp = spacy.load("en_core_web_sm")
def adj_dependcy_tree(argments, max_length):

    depend = []
    depend1 = []
    doc = nlp(str(argments))
    d = {}
    i = 0
    for (_, token) in enumerate(doc):
        if str(token) in d.keys():
            continue
        d[str(token)] = i
        i = i + 1

    for token in doc:
        depend.append((token.text, token.head.text))
        depend1.append((d[str(token)], d[str(token.head)]))

    ze = np.identity(max_length)
    for (i, j) in depend1:
        if i >= max_length or j >= max_length:
            continue
        ze[i][j] = 1

    return torch.FloatTensor(ze)

def dependency_tree(text, allens):
    allen = len(allens[-1])
    adjs = []
    for i in text:
        argments = str(i).strip('[').strip(']').replace("'","").replace(',','')
        adj = adj_dependcy_tree(argments, max_length=allen)
        adjs.append(torch.unsqueeze(adj, dim=0))

    adj_dependency_tree = torch.cat(adjs, dim=0)

    if torch.cuda.is_available():
        adj_dependency_tree = adj_dependency_tree.cuda()

    return adj_dependency_tree
