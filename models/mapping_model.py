import torch.nn as nn
import numpy as np
from training.loss import *
import torch

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, loss_function):
        nn.Module.__init__(self)
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = loss_function
    
    #@profile
    def forward(self, *args):
        batch = args[0]
        data = batch["input"]
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded

    def loss(self, *args):
        loss = self.loss_function.eval_loss(args[0])
        return loss
