import numpy as np
from modelFolder.RNN import LSTMTagger
from modelFolder.RNN import HIDDEN_DIM, EMBEDDING_DIM
import json
import torch

#BATCHSIZE個  很多字
#reshape 成所有 pair 的 list
# 2 句
# 每句 3 字
Loss = torch.tensor(
    [
        [[0.23,0.25],[0.23,0.25],[0.55,0.65]],
        [[-100,-100],[-100,-100],[-100,-100]]
    ])
A = torch.tensor(
    [
        [-100,-100],
        [-100,-100],
        [-100,-100],
        [-100,-100],
        [-100,-100],
        [0,1],
        [1,0],
        [0,1],
        [5,4]
    ])
#print(Loss)
print(Loss.view(-1,1))
print(A.view(-1,1))
#B = [[0,0]]

#print(A[B])