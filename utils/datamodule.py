import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

import os
import numpy as np
from . import load_json, CustomBatchSampler
import ot

class DataCollator:
    def __init__(self):
        pass

    def __call__(self, examples):
        ids, userId, itemId, labels = [], [], [], []
        isActive = True
        actives = []
        for _id, _userID, _itemID, _label, _isActive in examples:
            ids.append(_id)
            userId.append(_userID)
            itemId.append(_itemID)
            labels.append(_label)
            actives.append(_isActive)
            isActive = isActive and _isActive

        ids = torch.tensor(ids, dtype=torch.long)
        user_indices = torch.tensor(userId, dtype=torch.long)
        item_indices = torch.tensor(itemId, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            'id': ids,
            'user_indices': user_indices,
            'item_indices': item_indices,
            'labels': labels,
            'is_active': isActive
        }


class NeuMFDataset:
    def __init__(self):
        self.ID = []
        self.userID = []
        self.itemID = []
        self.labels = []
        self.isActive = []

    def add(self, ID, userID, itemID, label, isActive):
        self.ID.append(ID)
        self.userID.append(userID)
        self.itemID.append(itemID)
        self.labels.append(label)
        self.isActive.append(isActive)

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, i):
        return self.ID[i], self.userID[i], self.itemID[i], float(self.labels[i]), self.isActive[i]

    def is_active(self, i):
        return self.isActive[i]


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 dataset1: str = '',
                 dataset2: str = '',
                 dataset3: str = '',
                 active_percent: str = '0.05',
                 negative_sample_num: int = 99,
                 neg_per_pos: int = 4,
                 ):

        super().__init__()

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = os.path.join(data_dir, dataset1, dataset2, dataset3)
        self.data_save_dir = os.path.join(
            data_dir, dataset1, dataset2, dataset3, f'{active_percent}%active')
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.negative_sample_num = negative_sample_num
        self.neg_per_pos = neg_per_pos

    def load_dataset(self):
        # load dataset from file
        print(f'Loading data of {self.dataset1} / {self.dataset2} / {self.dataset3} / ...')
        print(self.data_save_dir)
        print(os.path.join(self.data_dir, "user2item.pkl"))

        user2item = pickle.load(
            open(os.path.join(self.data_dir, "user2item.pkl"), 'rb'))
        item2user = pickle.load(
            open(os.path.join(self.data_dir, "item2user.pkl"), 'rb'))
        user_list = pickle.load(
            open(os.path.join(self.data_dir, "user.pkl"), 'rb'))
        item_list = pickle.load(
            open(os.path.join(self.data_dir, "item.pkl"), 'rb'))

        # active vs static id load
        active_user_list = pickle.load(open(os.path.join(self.data_save_dir, "active.pkl"), 'rb'))
        static_user_list = pickle.load(open(os.path.join(self.data_save_dir, "static.pkl"), 'rb'))
        
        dis_half2half = self.optimal_find_half_dis(user2item, item_list, static_user_list)
        static2active_v2 = self.find_dis2adv_v2( user2item, item_list, active_user_list, static_user_list)

        print("Load succeed!")

        self.raw_datasets = {
            'user2item': user2item,
            'item2user': item2user,
            'user': user_list,
            'item': item_list,
            'active_users': list(active_user_list),
            'static_users': list(static_user_list),
            "dis_half2half": dis_half2half,
            "static2active_v2": static2active_v2,
        }

    def optimal_find_half_dis(self, user2item: dict, item_list: list, static_users: list):
        print("optimal find half dis...")
        # if True:
        if not os.path.exists(os.path.join(self.data_save_dir, "half2half_dis.pkl")):
            half2half = {}
            sorted_user_interactions = sorted(
                user2item.items(), key=lambda x: len(x[1]), reverse=True)
            sorted_users = [
                user for user, interactions in sorted_user_interactions if user in static_users]
            half_user_num = len(sorted_users) // 2
            if len(sorted_users) % 2 == 0:
                half_users_1 = sorted_users[:half_user_num]
                half_users_2 = sorted_users[half_user_num:]
            else:
                # 多的一个最中间的user和自己对应
                half2half[half_user_num] = half_user_num
                half_users_1 = sorted_users[:half_user_num]
                half_users_2 = sorted_users[half_user_num + 1:]
            one_hot_half_1 = []
            one_hot_half_2 = []
            for user, interactions in tqdm(user2item.items()):
                if user in half_users_1:
                    one_hot_half_1.append(
                        [0 if i not in interactions else 1 for i in item_list])
                elif user in half_users_2:
                    one_hot_half_2.append(
                        [0 if i not in interactions else 1 for i in item_list])
            a, b = ot.unif(half_user_num), ot.unif(half_user_num)
            one_hot_half_1 = np.array(one_hot_half_1)
            one_hot_half_2 = np.array(one_hot_half_2)
            print("Calculate dist...")
            M = ot.dist(one_hot_half_2, one_hot_half_1)
            print("Calculate transfer matrix...")
            P = ot.emd(b, a, M, numItermax=10000000)
            P = np.argmax(P, axis=1)
            print("OT find succeed!!!")
            for i in range(half_user_num):
                half2half[half_users_2[i]] = half_users_1[P[i]]
                half2half[half_users_1[P[i]]] = half_users_2[i]
            for user in sorted_users:
                if user not in half2half:
                    half2half[user] = user
            pickle.dump(half2half, open(os.path.join(
                self.data_save_dir, "half2half_dis.pkl"), 'wb'))
        else:
            half2half = pickle.load(
                open(os.path.join(self.data_save_dir, "half2half_dis.pkl"), 'rb'))
        return half2half


    def find_dis2adv_v2(self, user2item: dict, item_list: list, active_user_list: list, static_user_list: list):
        print("find dis2adv v2...")
        if not os.path.exists(os.path.join(self.data_save_dir, "static2active_v2.pkl")):
            static2active_v2 = {}
            for dis_u in static_user_list:
                intersect_adv = []
                for adv_u in active_user_list:
                    intersect_num = set(user2item[dis_u]).intersection(
                        user2item[adv_u])
                    intersect_adv.append([adv_u, intersect_num])

                intersect_adv.sort(key=lambda x: x[1], reverse=True)
                static2active_v2[dis_u] = intersect_adv[0][0]

            pickle.dump(static2active_v2, open(os.path.join(
                self.data_save_dir, "static2active_v2.pkl"), 'wb'))
        else:
            static2active_v2 = pickle.load(
                open(os.path.join(self.data_save_dir, "static2active_v2.pkl"), 'rb'))

        return static2active_v2

    def prepare_dataset(self):
        # process dataset into samples
        self.processed_dataset = {}

        print(f"loading from {os.path.join(self.data_save_dir)} ")

        if not os.path.exists(os.path.join(self.data_save_dir, 'test_nega.pkl')):
            vali_data = {}
            train_data = {}
            test_data = {}
            test_neg_data = {}
            vali_neg_data = {}
            train_neg_data = {}

            for user in tqdm(self.raw_datasets['user2item']):
                items = self.raw_datasets['user2item'][user]
                items = items.copy()
                if len(items) >= 2:
                    np.random.shuffle(items)
                    vali_data[user] = items[0]
                    test_data[user] = items[1]
                    train_data[user] = items[2:]
                elif len(items) == 1:
                    vali_data[user] = items[0]
                    test_data[user] = items[0]
                    train_data[user] = [items[0]]
                else:
                    vali_data[user] = 0
                    test_data[user] = 0
                    train_data[user] = [0]
                negative_item_pool = list(set(self.raw_datasets['item']) - set(items))
                np.random.shuffle(negative_item_pool)
                train_neg_num = len(train_data[user]) * self.neg_per_pos
                test_neg_data[user] = negative_item_pool[:self.negative_sample_num]
                vali_neg_data[user] = negative_item_pool[self.negative_sample_num: 2 *
                                                         self.negative_sample_num]
                train_neg_data[user] = negative_item_pool[2 *
                                                          self.negative_sample_num: 2*self.negative_sample_num + train_neg_num]

            pickle.dump(train_data, open(os.path.join(
                self.data_save_dir, "train.pkl"), 'wb'))
            pickle.dump(test_data, open(os.path.join(
                self.data_save_dir, "test.pkl"), 'wb'))
            pickle.dump(vali_data, open(os.path.join(
                self.data_save_dir, "vali.pkl"), 'wb'))
            pickle.dump(train_neg_data, open(os.path.join(
                self.data_save_dir, "train_nega.pkl"), 'wb'))
            pickle.dump(vali_neg_data, open(os.path.join(
                self.data_save_dir, "vali_nega.pkl"), 'wb'))
            pickle.dump(test_neg_data, open(os.path.join(
                self.data_save_dir, "test_nega.pkl"), 'wb'))

        else:
            train_data = pickle.load(
                open(os.path.join(self.data_save_dir,  "train.pkl"), 'rb'))
            vali_data = pickle.load(
                open(os.path.join(self.data_save_dir, "vali.pkl"), 'rb'))
            test_data = pickle.load(
                open(os.path.join(self.data_save_dir, "test.pkl"), 'rb'))
            test_neg_data = pickle.load(
                open(os.path.join(self.data_save_dir, "test_nega.pkl"), 'rb'))
            vali_neg_data = pickle.load(
                open(os.path.join(self.data_save_dir, "vali_nega.pkl"), 'rb'))
            train_neg_data = pickle.load(
                open(os.path.join(self.data_save_dir, "train_nega.pkl"), 'rb'))

        self.processed_dataset['train'] = self.process_dict_to_dataset(
            train_data, train_neg_data, 'train')
        self.processed_dataset['test'] = self.process_dict_to_dataset(
            test_data, test_neg_data, 'test')
        self.processed_dataset['dev'] = self.process_dict_to_dataset(
            vali_data, vali_neg_data, 'dev')
        assert len(self.processed_dataset['test']) % (
            1 + self.negative_sample_num) == 0, "test dataset size error"
        assert len(self.processed_dataset['dev']) % (
            1 + self.negative_sample_num) == 0, "dev dataset size error"

        print("prepare succeed!!!")

    def get_num_info(self):
        # info for model initialization
        user_num = len(self.raw_datasets['user'])
        item_num = len(self.raw_datasets['item'])
        active_user_list = self.raw_datasets['active_users']
        static_user_list = self.raw_datasets['static_users']
        return user_num, item_num, active_user_list, static_user_list

    def get_dis_half(self):
        return self.raw_datasets['dis_half2half']

    def get_edge_info(self, ):
        user_list, item_list = [], []
        for _, _userID, _itemID, _label, _ in self.processed_dataset['train']:
            if _label == 1:
                user_list.append(_userID)
                item_list.append(_itemID)
        return user_list, item_list

    def process_dict_to_dataset(self, dataset_dict, neg_state_dict, mode=''):
        id = 0
        new_dataset = NeuMFDataset()
        for user, items in dataset_dict.items():

            isActive = True if user in self.raw_datasets['active_users'] else False
            if mode == 'train':
                for item in items:
                    new_dataset.add(id, user, item, 1, isActive=isActive)
                    id += 1

                for neg_item in neg_state_dict[user]:
                    new_dataset.add(id, user, neg_item, 0, isActive=isActive)
                    id += 1

            else:
                new_dataset.add(id, user, items, 1, isActive=isActive)
                id += 1
                for neg_item in neg_state_dict[user]:
                    new_dataset.add(id, user, neg_item, 0, isActive=isActive)
                    id += 1

        return new_dataset

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.processed_dataset[mode],
            batch_sampler=CustomBatchSampler(
                self.processed_dataset[mode], batch_size, drop_last=False),
            collate_fn=DataCollator(),
            pin_memory=True,
            prefetch_factor=16,
            num_workers=1
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)
