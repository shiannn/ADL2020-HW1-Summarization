import torch
from torch.nn import Sigmoid
import torch.nn as nn
from multiprocessing import Pool, cpu_count
import json
import numpy as np
from rouge_score.rouge_scorer import RougeScorer

with open("predict.jsonl","r") as f:
    for line in f:
        print(line)
        a = json.loads(line)
        print(a)
        p = {a['id']: a}
        print(p)
        break
    """
    predicts = [json.loads(line) for line in f]
    predicts = {p['id']: p for p in predicts}
    print(predicts)
    """