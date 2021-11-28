import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd

class LabelBank(object):
    def __init__(self,thre=0.8):
        super(LabelBank, self).__init__()
        self.pseudo_labels=[]

    @classmethod
    def update_label()
