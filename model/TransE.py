import math
import random
import time

import torch
from torch import nn
from torch.nn import init

from utils.DataLoader import DataLoader
from utils.draw import draw


class TransE(nn.Module):
    """TransE Model"""
    def __init__(self, device, entity_size, rel_size, embed_dim, *args):
        super(TransE, self).__init__()
        self.device = device
        self.entity_size = entity_size
        self.rel_size = rel_size
        self.embed_dim = embed_dim
        self.entity_embedding = nn.Embedding(entity_size, embed_dim).to(device)
        self.rel_embedding = nn.Embedding(rel_size, embed_dim).to(device)

    def normalize_layer(self):
        w1 = self.entity_embedding.weight.detach()
        self.entity_embedding.weight.data = w1/w1.norm(dim=-1, keepdim=True)
        w2 = self.rel_embedding.weight.detach()
        self.rel_embedding.weight.data = w2/w2.norm(dim=-1, keepdim=True)

    def distance(self, h,r,t,ord):
        return (h + r - t).norm(p=ord,dim=-1,keepdim=True)

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
        l = (2 + d1 - d2).clamp(min=0)
        return l
    
    def evaluate(self, h, r, t):
        h_embed = self.entity_embedding(t)
        r_embed = self.rel_embedding(r)
        target_embed = self.entity_embedding(t)
        t_embed = self.entity_embedding.weight.data.detach()
        target = self.distance(h_embed, r_embed, target_embed, 1)
        out = self.distance(h_embed, r_embed, t_embed, 1)
        result = (out < target).sum()
        return result.item()



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = DataLoader(device, dataset='WN18')
    loader.load_all()
    loader.preprocess(1, init=False)
    loader.setup_sampling_map()
    
    entity_dim = 20

    model = TransE(device, entity_size=loader.entity_size, 
                    rel_size=loader.relation_size, embed_dim=entity_dim)

    init.uniform_(model.entity_embedding.weight,-6.0/math.sqrt(entity_dim), 6.0/math.sqrt(entity_dim))
    init.uniform_(model.rel_embedding.weight,-6.0/math.sqrt(entity_dim), 6.0/math.sqrt(entity_dim))
    
    # for params in model.parameters():
    #     init.normal_(params, mean=0, std=1)

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    batch_size = 500
    epoch_num = 30

    display_l = []

    for epoch in range(epoch_num):
        start = time.time()
        l = 0.0
        cnt = 0
        model.normalize_layer()
        dataiter = loader.get_dataiter_('train', batch_size, True)
        for h,r,t,h_hat,t_hat in dataiter:
            if random.random() < 0.5:
                d = model(h,r,t,h_hat,t)
            else:
                d = model(h,r,t,h,t_hat)
            cnt += d.shape[0]
            l += loss(d, torch.zeros_like(d))*d.shape[0]
        # print(time.time()-start)
        l.backward()
        optimizer.step()
        display_l.append(l.item()/cnt)
        print("{} time: {} : {}".format(epoch, time.time()-start, l.item()))
        draw(display_l)

    valid = []
    for h,r,t in loader.get_dataiter_('valid', 1, True):
        valid.append(model.evaluate(h,r,t))
    print(sum(valid)/len(valid))

    test = []
    for h,r,t in loader.get_dataiter_('test', 1, True):
        test.append(model.evaluate(h,r,t))
    print(sum(test)/len(test))
