import sys

import torch
from torch.utils import data as Data

from utils.DataLoader import DataLoader


class DataSampler(DataLoader):
    def __init__(self, *args, **kw):
        super(DataSampler, self).__init__(*args, **kw)
        self.ukn_rel = -1
        self.ukn_train_triple = []
        self.ukn_valid_triple = []
        self.ukn_test_triple = []
        self.ukn_train_triple_size = 0
        self.ukn_valid_triple_size = 0
        self.ukn_test_triple_size = 0
        pass

    def set_ukn(self, ukn_rel:int):
        print('Setting Ukn: %d'% ukn_rel)
        self.ukn_rel = ukn_rel
        
        self.ukn_train_triple = []
        triple_tmp = []
        for triple in self.train_triple:
            if triple[1] == self.ukn_rel:
                self.ukn_train_triple.append(triple)
            else:
                triple_tmp.append(triple)
        self.train_triple = triple_tmp
        self.train_triple_size = len(self.train_triple)
        self.ukn_train_triple_size = len(self.ukn_train_triple)
        
        self.ukn_valid_triple = []
        triple_tmp = []
        for triple in self.valid_triple:
            if triple[1] == self.ukn_rel:
                self.ukn_valid_triple.append(triple)
            else:
                triple_tmp.append(triple)
        self.valid_triple = triple_tmp
        self.valid_triple_size = len(self.valid_triple)
        self.ukn_valid_triple_size = len(self.ukn_valid_triple)
        
        self.ukn_test_triple = []
        triple_tmp = []
        for triple in self.test_triple:
            if triple[1] == self.ukn_rel:
                self.ukn_test_triple.append(triple)
            else:
                triple_tmp.append(triple)
        self.test_triple = triple_tmp
        self.test_triple_size = len(self.test_triple)
        self.ukn_test_triple_size = len(self.ukn_test_triple)

    def get_ukn_dataiter_(self, mode='train', batch_size=10, shuffle=False, num_workers=0):
        if mode == 'train':
            dataset_h = torch.tensor([i[0] for i in self.ukn_train_triple], device=self.device)
            dataset_r = torch.tensor([i[1] for i in self.ukn_train_triple], device=self.device)
            dataset_t = torch.tensor([i[2] for i in self.ukn_train_triple], device=self.device)
            dataset_h_hat = torch.randint_like(dataset_h, high=self.entity_size, device=self.device)
            dataset_t_hat = torch.randint_like(dataset_t, high=self.entity_size, device=self.device)
            batch_num = self.ukn_train_triple_size // batch_size + 1
            for i in range(batch_num):
                yield dataset_h[i*batch_size : i*batch_size+batch_size], \
                      dataset_r[i*batch_size : i*batch_size+batch_size], \
                      dataset_t[i*batch_size : i*batch_size+batch_size], \
                      dataset_h_hat[i*batch_size : i*batch_size+batch_size], \
                      dataset_t_hat[i*batch_size : i*batch_size+batch_size]
        if mode == 'valid':
            dataset_h = torch.tensor([i[0] for i in self.ukn_valid_triple], device=self.device)
            dataset_r = torch.tensor([i[1] for i in self.ukn_valid_triple], device=self.device)
            dataset_t = torch.tensor([i[2] for i in self.ukn_valid_triple], device=self.device)
            for i in range(self.ukn_valid_triple_size):
                yield dataset_h[i], \
                      dataset_r[i], \
                      dataset_t[i]
        if mode == 'test':
            dataset_h = torch.tensor([i[0] for i in self.ukn_test_triple], device=self.device)
            dataset_r = torch.tensor([i[1] for i in self.ukn_test_triple], device=self.device)
            dataset_t = torch.tensor([i[2] for i in self.ukn_test_triple], device=self.device)
            for i in range(self.ukn_test_triple_size):
                yield dataset_h[i], \
                      dataset_r[i], \
                      dataset_t[i]
