import time

import torch
from nltk.corpus import wordnet as wn
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

model = TransE(device, entity_size=sampler.entity_size, 
                rel_size=sampler.relation_size, 
                embed_dim=entity_dim, dataset='WN18', margin=2)

model.set_ukn(ukn_rel)

model.load_net(model.model_name)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 1000
epoch_num = 0

display_l = []

for epoch in range(epoch_num):
    start = time.time()
    l = 0.0
    cnt = 0
    model.normalize_layer(False)
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

# test = []
# cnt = 0
# for h,r,t in sampler.get_dataiter_('test', 1, True):
#     test.append(model.evaluate(h,r,t))
#     if test[-1] < 10:
#         cnt += 1
# print(sum(test)/len(test))
# print(cnt*1.0/len(test))

# kn_heads = [i[0] for i in sampler.train_triple]
# kn_tails = [i[2] for i in sampler.train_triple]
# cnt = 0
# for ukn_triple in sampler.ukn_train_triple:
#     flag1 = ukn_triple[0] in kn_heads
#     flag2 = ukn_triple[2] in kn_tails
#     print(ukn_triple)
#     print(flag1)
#     print(flag2)
#     if flag1: cnt+=1
#     if flag2: cnt+=1
# print(cnt)

# v1, i1 = model.search_with_head_rel(0,0)
# v2, i2 = model.search_with_head_rel(0,1)

# print(v1)
# print(v2)
# print(i1)
# print(i2)

# print(torch.cat([v1,v2],dim=0))

def get_def(word):
    t = wn.synset(word).lemma_names()[0]
    return t


# print(sampler.relation_map.)
print(len(sampler.all_triple))

for triple in sampler.train_triple[1:]:
    print('triple: ', end='')
    print(triple)
    print(sampler.entity_reverse_map[triple[0]], end=' ')
    print(sampler.relation_reverse_map[triple[1]], end=' ')
    print(sampler.entity_reverse_map[triple[2]])
    # print(model.beam_search_with_head(triple[0],ukn_rel=0,k=10))
    base = 1
    flag = False
    while not flag and base<6:
        print('base %d:'% base)
        path = model.find_path(triple[0], triple[2], 0, 100, base)
        for i in path:
            if i[1][-1][1] == triple[2]:
                flag = True
                print(sampler.entity_reverse_map[triple[0]], end=' ')
                for p in i[1]:
                    print(sampler.relation_reverse_map[p[0]], end=' ')
                    print(sampler.entity_reverse_map[p[1]], end=' ')
                print()
                head = triple[0]
                for p in i[1]:
                    # print(head, *p)
                    if(sampler.slow_check((head, *p))):
                        print('check '+str(head))
                    head = p[1]
        base += 2
