import math
import time

import torch
from torch import nn
from torch.nn import init

from utils.DataLoader import DataLoader


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

    def forward(self, heads, relations, tails):
        # entities: (batch_size)
        # relations: (batch_size)

        # h_embed: (batch_size, embed_dim)
        # r_embed: (batch_size, embed_dim)
        # t_embed: (batch_size, embed_dim)
        h_embed = self.entity_embedding(heads)
        r_embed = self.rel_embedding(relations)
        t_embed = self.entity_embedding(tails)
        d = h_embed + r_embed - t_embed
        return d


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = DataLoader(device, dataset='WN18')
    loader.load_all()
    loader.preprocess(1, init=False)
    loader.setup_sampling_map()
    
    entity_dim = 5

    model = TransE(device, entity_size=loader.entity_size, 
                    rel_size=loader.relation_size, embed_dim=entity_dim)

    init.uniform_(model.entity_embedding.weight,-6.0/math.sqrt(entity_dim), 6.0/math.sqrt(entity_dim))
    init.uniform_(model.rel_embedding.weight,-6.0/math.sqrt(entity_dim), 6.0/math.sqrt(entity_dim))
    
    # for params in model.parameters():
    #     init.normal_(params, mean=0, std=1)

    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    batch_size = 1000
    epoch_num = 100

    for epoch in range(epoch_num):
        start = time.time()
        l = 0.0
        cnt = 0
        model.normalize_layer()
        dataiter = loader.get_dataiter('train', batch_size, False)
        for h,r,t in dataiter:
            d = model(h,r,t)
            cnt += d.shape[0]
            l += loss(d, torch.zeros_like(d))*d.shape[0]
        print(time.time()-start)
        l.backward()
        optimizer.step()
        print("{} time: {} : {}".format(epoch, time.time()-start, l.item()))
