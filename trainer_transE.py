import math
import random
import time

import torch
from torch import nn
from torch.nn import init

from model.TransE import TransE
from utils.DataLoader import DataLoader
from utils.draw import draw

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(device, dataset='FB15k')
loader.load_all()
loader.preprocess(1, init=False)
loader.setup_sampling_map()

entity_dim = 50
load_previous = False

model = TransE(device, entity_size=loader.entity_size, 
                rel_size=loader.relation_size, embed_dim=entity_dim, dataset='FB15k', margin=5)

# init.uniform_(model.entity_embedding.weight,-6.0/math.sqrt(entity_dim), 6.0/math.sqrt(entity_dim))
# init.uniform_(model.rel_embedding.weight,-6.0/math.sqrt(entity_dim), 6.0/math.sqrt(entity_dim))

if not load_previous:
    for params in model.parameters():
        init.normal_(params, mean=0, std=1)
else:
    model.load_net(model.model_name)

# loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 1000
epoch_num = 400

display_l = []

for epoch in range(epoch_num):
    start = time.time()
    l = 0.0
    cnt = 0
    model.normalize_layer(True if epoch == 0 and not load_previous else False)
    optimizer.zero_grad()
    # model.entity_embedding.weight.data.zero_grad()
    # model.rel_embedding.weight.data.zero_grad()
    dataiter = loader.get_dataiter_('train', batch_size, True)
    for h,r,t,h_hat,t_hat in dataiter:
        # if random.random() < 0.5:
        #     d = model(h,r,t,h_hat,t)
        # else:
        #     d = model(h,r,t,h,t_hat)
        d = model(h,r,t,h,t_hat)
        cnt += d.shape[0]
        l += d.sum()
        # break
        d.sum().backward()
    # print(time.time()-start)
    # l.backward()
    optimizer.step()
    display_l.append(l.item()/cnt)
    if (epoch+1) % 1 == 0:
        print("{} time: {} : {}".format(epoch+1, time.time()-start, l.item()))
        draw(display_l)
    if (epoch+1) % 50 == 0 or epoch==0:
        valid = []
        cnt = 0
        for h,r,t in loader.get_dataiter_('valid', 1, True):
            valid.append(model.evaluate(h,r,t))
            if valid[-1] < 10:
                cnt += 1
        print(sum(valid)/len(valid))
        print(cnt*1.0/len(valid))
        model.save_net(model.model_name)

test = []
cnt = 0
for h,r,t in loader.get_dataiter_('test', 1, True):
    test.append(model.evaluate(h,r,t))
    if test[-1] < 10:
        cnt += 1
print(sum(test)/len(test))
print(cnt*1.0/len(test))

# train = []
# cnt = 0
# for h,r,t,_,_ in loader.get_dataiter_('train', 1, True):
#     train.append(model.evaluate(h,r,t))
#     if tra
# print(sum(train)/len(train))


model.save_net(model.model_name)
