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

    def set_ukn(self, ukn_rel):
        self.model_name += '_'+str(ukn_rel)

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

    def search_with_head_rel(self, h, r, k=2):
        h_embed = self.entity_embedding(torch.tensor([h], device=self.device)).detach()
        r_embed = self.rel_embedding(torch.tensor([r], device=self.device)).detach()
        t_embed = self.entity_embedding.weight.data.detach()
        out = self.distance(h_embed, r_embed, t_embed, 1).reshape((-1))
        values, indices = out.topk(k, largest=False)
        indices = indices.tolist()
        if h in indices:
            values, indices = out.topk(k+1, largest=False)
            indices = indices.tolist()
            loc = indices.index(h)
            values = values[torch.arange(values.size(0))!=loc]
            indices = indices[:loc]+indices[loc+1:]
        return values, indices

    def beam_search_with_head(self, h, ukn_rel, k=2):
        values = []
        pairs = []
        for rel in range(self.rel_size):
            if rel == ukn_rel:
                continue
            value, indice = self.search_with_head_rel(h, rel, k)
            values.append(value)
            pairs.extend([(rel, i) for i in indice])
        values = torch.cat(values, dim=0)
        out_values, out_pairs = values.topk(k, largest=False)
        out_values = out_values.tolist()
        out_pairs = out_pairs.tolist()
        return out_values, [pairs[i] for i in out_pairs]

    # def find_path(self, h, accumulated_weight, t, ukn_rel, k=2, max_depth=2):
    #     if max_depth==0:
    #         return []
    #     values = torch.tensor([],device=self.device)
    #     pairs = []
    #     path = []
    #     idx = 0
    #     v, p = self.beam_search_with_head(h,ukn_rel,k)
    #     for i in range(len(p)):
    #         v2, p2 = self.beam_search_with_head(p[i][1], ukn_rel, k)
    #         for j in range(k):
    #             tmp = (v[i]+v2[j], (p[i],p2[j]))
    #             path.append(tmp)
    #     return path

    def find_path(self, h, t, ukn_rel, k=10, max_depth=2, history=[]):
        # print(len(history))
        if len(history) == 0:
            v, p = self.beam_search_with_head(h,ukn_rel,k)
            for i in range(len(p)):
                tmp = (v[i], [p[i]])
                history.append(tmp)
            return self.find_path(h, t, ukn_rel, k, max_depth, history)
        else:
            flag = False
            new_history = []
            for log in history:
                # print(log)
                if len(log[1]) >= max_depth:
                    new_history.append(log)
                elif log[1][-1][1] == t:
                    new_history.append(log)
                else:
                    flag = True
                    v, p = self.beam_search_with_head(log[1][-1][1], ukn_rel, k)
                    for i in range(len(p)):
                        tmp = (log[0]+v[i], log[1]+[p[i]])
                        new_history.append(tmp)
            sorted(new_history, key=lambda t: t[0])
            if not flag:
                return new_history[:k]
            else:
                return self.find_path(h, t, ukn_rel, k, max_depth, new_history[:k])


    def save_net(self, save_path='WN18_30'):
        torch.save(self.state_dict(), save_path+'.params')
        print('model saved')
    
    def load_net(self, load_path='WN18_30'):
        self.load_state_dict(torch.load(load_path+'.params', map_location=self.device))
        print('model loaded')
