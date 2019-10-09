import math
import random
import time

import torch
from torch import nn
from torch.nn import init

from model.TransE import TransE
from utils.DataLoader import DataLoader
from utils.draw import draw

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
    model.normalize_layer(True if epoch == 0 else False)
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
