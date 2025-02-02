import collections

import torch
from torch.utils import data as Data


class DataLoader(object):
    def __init__(self, device, dataset='WN18', *args):
        self.device = device
        self.train_path = './data/{}/train.txt'.format(dataset)
        self.valid_path = './data/{}/valid.txt'.format(dataset)
        self.test_path = './data/{}/test.txt'.format(dataset)
        self.entity_map_path = './data/{}/entity.map'.format(dataset)
        self.relation_map_path = './data/{}/relation.map'.format(dataset)
        self.train_list = [] # Will be deleted
        self.valid_list = [] # Will be deleted
        self.test_list = [] # Will be deleted
        self.entity_map = {}
        self.relation_map = {}
        self.entity_size = 0
        self.relation_size = 0
        self.train_triple = []
        self.valid_triple = []
        self.test_triple = []
        self.head_relation_to_tail = [] # l[r][h]=t
        self.tail_relation_to_head = [] # l[r][t]=h
        self.train_triple_size = 0
        self.valid_triple_size = 0
        self.test_triple_size = 0
        
    def load_all(self):
        self.train_list = []
        with open(self.train_path, 'r') as f:
            lines = f.readlines()
        self.train_list = [line.split() for line in lines]
        print('Trainset size: {}'.format(len(self.train_list)))

        self.valid_list = []
        with open(self.valid_path, 'r') as f:
            lines = f.readlines()
        self.valid_list = [line.split() for line in lines]
        print('Validset size: {}'.format(len(self.valid_list)))
        
        self.test_list = []
        with open(self.test_path, 'r') as f:
            lines = f.readlines()
        self.test_list = [line.split() for line in lines]
        print('Testset size: {}'.format(len(self.test_list)))
    
    def _counter_filter(self, raw_dataset, count=1):
        counter = collections.Counter([tk for tk in raw_dataset])
        counter = dict(filter(lambda x: x[1] >= count, counter.items()))
        return counter
    
    def setup_sampling_map(self):
        print('Setting up sampling map')
        self.head_relation_to_tail = [{}]*self.relation_size
        self.tail_relation_to_head = [{}]*self.relation_size
        print('Adding sampling dataset')
        for (head, relation, tail) in self.train_triple:
            if head in self.head_relation_to_tail[relation].keys():
                self.head_relation_to_tail[relation][head].append(tail)
            else:
                self.head_relation_to_tail[relation][head] = [tail]

            if tail in self.tail_relation_to_head[relation].keys():
                self.tail_relation_to_head[relation][tail].append(head)
            else:
                self.tail_relation_to_head[relation][tail] = [head]
        print('Finished setting up sampling map')

    def preprocess(self, filter_occurance=5, init=False):
        '''Preprocess the dataset.

        Parameters
        ----------
        filter_occurance : int
            Only entities that occur no fewer than 'filter_occurance' will be included 
            (occurring in either head or tail is qualified).
        init : bool, default False
            Whether to recreate entity2idx map and relation2idx map.
        '''
        all_list = [*self.train_list,*self.valid_list,*self.test_list]
        entity_list = []
        relation_list = []
        for triple in all_list:
            entity_list.append(triple[0])
            relation_list.append(triple[1])
            entity_list.append(triple[2])
        entity_counter = self._counter_filter(entity_list, filter_occurance)
        relation_counter = self._counter_filter(relation_list, 1)
        if init:
            for (i, entity) in enumerate(entity_counter.keys()):
                self.entity_map[entity] = i
            for (i, relation) in enumerate(relation_counter.keys()):
                self.relation_map[relation] = i
            with open(self.entity_map_path, 'w') as f:
                f.write(str(self.entity_map))
            with open(self.relation_map_path, 'w') as f:
                f.write(str(self.relation_map))
        else:
            print('Reading {}'.format(self.entity_map_path))
            with open(self.entity_map_path, 'r') as f:
                self.entity_map = eval(f.read())
            print('Reading {}'.format(self.relation_map_path))
            with open(self.relation_map_path, 'r') as f:
                self.relation_map = eval(f.read())
        self.entity_reverse_map = dict(zip(self.entity_map.values(), self.entity_map.keys()))
        self.relation_reverse_map = dict(zip(self.relation_map.values(), self.relation_map.keys()))
        
        self.entity_size = len(self.entity_map.keys())
        self.relation_size = len(self.relation_map.keys())
        
        print('Entity_size: {}'.format(self.entity_size))
        print('Relation_size: {}'.format(self.relation_size))

        self.train_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[2]]) 
                                for i in self.train_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.valid_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[2]]) 
                                for i in self.valid_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.test_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[2]]) 
                                for i in self.test_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.train_triple_size = len(self.train_triple)
        self.valid_triple_size = len(self.valid_triple)
        self.test_triple_size = len(self.test_triple)
        self.all_triple = [*self.train_triple, *self.valid_triple, *self.test_triple]

    def del_raw_data(self):
        del self.train_list
        del self.valid_list
        del self.test_list
            
    def slow_check(self, triple):
        return triple in self.all_triple

    def check_with_h_r(self, h, r, t):
        if self.head_relation_to_tail[r][h] != None and t in self.head_relation_to_tail[r][h]:
            return True
        else:
            return False

    def check_with_t_r(self, h, r, t):
        if self.tail_relation_to_head[r][t] != None and h in self.tail_relation_to_head[r][t]:
            return True
        else:
            return False
    
    def get_t_list_with_h_r(self, h, r):
        if self.head_relation_to_tail[r][h] == None:
            return []
        else:
            return self.head_relation_to_tail[r][h]
    
    def get_h_list_with_r_t(self, r, t):
        if self.tail_relation_to_head[r][t] == None:
            return []
        else:
            return self.tail_relation_to_head[r][t]
    
    # def get_dataiter(self, mode='train', batch_size=10, shuffle=False, num_workers=0):
    #     '''deprecated'''
    #     if mode=='train':       
    #         dataset = Data.TensorDataset(torch.tensor([i[0] for i in self.train_triple], device=self.device), 
    #                                     torch.tensor([i[1] for i in self.train_triple], device=self.device),
    #                                     torch.tensor([i[2] for i in self.train_triple], device=self.device))
    #         data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #     if mode=='val':
    #         dataset = Data.TensorDataset(torch.tensor([i[0] for i in self.valid_triple], device=self.device), 
    #                                     torch.tensor([i[1] for i in self.valid_triple], device=self.device),
    #                                     torch.tensor([i[2] for i in self.valid_triple], device=self.device))
    #         data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    #     if mode=='test':
    #         dataset = Data.TensorDataset(torch.tensor([i[0] for i in self.test_triple], device=self.device), 
    #                                     torch.tensor([i[1] for i in self.test_triple], device=self.device),
    #                                     torch.tensor([i[2] for i in self.test_triple], device=self.device))
    #         data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    #     return data_iter
    
    def get_dataiter_(self, mode='train', batch_size=10, shuffle=False, num_workers=0):
        if mode == 'train':
            dataset_h = torch.tensor([i[0] for i in self.train_triple], device=self.device)
            dataset_r = torch.tensor([i[1] for i in self.train_triple], device=self.device)
            dataset_t = torch.tensor([i[2] for i in self.train_triple], device=self.device)
            dataset_h_hat = torch.randint_like(dataset_h, high=self.entity_size, device=self.device)
            dataset_t_hat = torch.randint_like(dataset_t, high=self.entity_size, device=self.device)
            batch_num = self.train_triple_size // batch_size + 1
            for i in range(batch_num):
                yield dataset_h[i*batch_size : i*batch_size+batch_size], \
                      dataset_r[i*batch_size : i*batch_size+batch_size], \
                      dataset_t[i*batch_size : i*batch_size+batch_size], \
                      dataset_h_hat[i*batch_size : i*batch_size+batch_size], \
                      dataset_t_hat[i*batch_size : i*batch_size+batch_size]
        if mode == 'valid':
            dataset_h = torch.tensor([i[0] for i in self.valid_triple], device=self.device)
            dataset_r = torch.tensor([i[1] for i in self.valid_triple], device=self.device)
            dataset_t = torch.tensor([i[2] for i in self.valid_triple], device=self.device)
            for i in range(self.valid_triple_size):
                yield dataset_h[i], \
                      dataset_r[i], \
                      dataset_t[i]
        if mode == 'test':
            dataset_h = torch.tensor([i[0] for i in self.test_triple], device=self.device)
            dataset_r = torch.tensor([i[1] for i in self.test_triple], device=self.device)
            dataset_t = torch.tensor([i[2] for i in self.test_triple], device=self.device)
            for i in range(self.test_triple_size):
                yield dataset_h[i], \
                      dataset_r[i], \
                      dataset_t[i]

    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(device, dataset='FB15k')
    loader.load_all()
    loader.preprocess(1, init=False)
    loader.setup_sampling_map()

    import time
    start = time.time()
    for i in range(10000):
        if loader.check_with_h_r(1, 2, i):
            print('hit: {}'.format(i))
    end = time.time()
    print(end - start)
