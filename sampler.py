import time

import torch
from torch.nn import init

from model.TransE import TransE
from utils.DataSampler import DataSampler
from utils.draw import draw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sampler = DataSampler(device, dataset='WN18')
sampler.load_all()
sampler.preprocess(1, init=False)
sampler.setup_sampling_map()
sampler.del_raw_data()

ukn_rel = 0

sampler.set_ukn(ukn_rel)

print('ukn_train_set: %d'%len(sampler.ukn_train_triple))
print('ukn_valid_set: %d'%len(sampler.ukn_valid_triple))
print('ukn_test_set: %d'%len(sampler.ukn_test_triple))
print('train_set: %d'%len(sampler.train_triple))
print('valid_set: %d'%len(sampler.valid_triple))
print('test_set: %d'%len(sampler.test_triple))


entity_dim = 20
load_previous = True

model = TransE(device, entity_size=sampler.entity_size, 
                rel_size=sampler.relation_size, 
                embed_dim=entity_dim, dataset='WN18', margin=2)

model.set_ukn(ukn_rel)

if not load_previous:
    for params in model.parameters():
        init.normal_(params, mean=0, std=1)
else:
    model.load_net(model.model_name)

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
    dataiter = sampler.get_dataiter_('train', batch_size, True)
    for h,r,t,h_hat,t_hat in dataiter:
        d = model(h,r,t,h,t_hat)
        cnt += d.shape[0]
        l += d.sum()
        d.sum().backward()
    optimizer.step()
    display_l.append(l.item()/cnt)
    if (epoch+1) % 1 == 0:
        print("{} time: {} : {}".format(epoch+1, time.time()-start, l.item()))
        draw(display_l)
    if (epoch+1) % 50 == 0 or epoch==0:
        valid = []
        cnt = 0
        for h,r,t in sampler.get_dataiter_('valid', 1, True):
            valid.append(model.evaluate(h,r,t))
            if valid[-1] < 10:
                cnt += 1
        print(sum(valid)/len(valid))
        print(cnt*1.0/len(valid))
        model.save_net(model.model_name)

test = []
cnt = 0
for h,r,t in sampler.get_dataiter_('test', 1, True):
    test.append(model.evaluate(h,r,t))
    if test[-1] < 10:
        cnt += 1
print(sum(test)/len(test))
print(cnt*1.0/len(test))

model.save_net(model.model_name)
