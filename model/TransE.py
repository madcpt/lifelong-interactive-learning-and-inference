import math
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from utils.DataLoader import DataLoader
from utils.draw import draw

sys.path.append('..')


class TransE(nn.Module):
    """TransE Model"""
    def __init__(self, device, entity_size, rel_size, embed_dim, dataset='WN18', margin=2, *args):
        super(TransE, self).__init__()
        self.device = device
        self.entity_size = entity_size
        self.rel_size = rel_size
        self.embed_dim = embed_dim
        self.entity_embedding = nn.Embedding(entity_size, embed_dim).to(device)
        self.rel_embedding = nn.Embedding(rel_size, embed_dim).to(device)
        self.distfn = nn.PairwiseDistance(1)
        self.model_name = dataset+'_'+str(self.embed_dim)
        self.margin = margin

    def normalize_layer(self, init = False):
        w1 = self.entity_embedding.weight.detach()
        self.entity_embedding.weight.data = w1/w1.norm(dim=-1, keepdim=True)
        if init:
            w2 = self.rel_embedding.weight.detach()
            self.rel_embedding.weight.data = w2/w2.norm(dim=-1, keepdim=True)

    def distance(self, h,r,t,ord):
        return (h + r - t).norm(p=ord,dim=-1,keepdim=True)
        # return self.distfn(h+r, t)

    def forward(self, heads, relations, tails, h_hat, t_hat):
        # entities: (batch_size)
        # relations: (batch_size)

        # h_embed: (batch_size, embed_dim)
        # r_embed: (batch_size, embed_dim)
        # t_embed: (batch_size, embed_dim)
        h_embed = self.entity_embedding(heads)
        r_embed = self.rel_embedding(relations)
        t_embed = self.entity_embedding(tails)
        h_hat_embed = self.entity_embedding(h_hat)
        t_hat_embed = self.entity_embedding(t_hat)
        d1 = self.distance(h_embed, r_embed, t_embed, 1)
        d2 = self.distance(h_hat_embed, r_embed, t_hat_embed, 1)
        # l = (2 + d1 - d2)
        l = (self.margin + d1 - d2).clamp_min(0)
        return l
    
    def evaluate(self, h, r, t):
        h_embed = self.entity_embedding(h)
        r_embed = self.rel_embedding(r)
        target_embed = self.entity_embedding(t)
        t_embed = self.entity_embedding.weight.data.detach()
        target = self.distance(h_embed, r_embed, target_embed, 1)
        out = self.distance(h_embed, r_embed, t_embed, 1)
        result = (out < target).sum()
        return result.item()

    def save_net(self, save_path='WN18_30'):
        torch.save(self.state_dict(), save_path+'.params')
        print('model saved')
    
    def load_net(self, load_path='WN18_30'):
        self.load_state_dict(torch.load(load_path+'.params', map_location=self.device))
        print('model loaded')
