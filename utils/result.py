import os
import time
import ujson as json
from utils import append_new_line
import torch
from utils.metrics import *
import pandas as pd

metric_K = 10

class Result:
    def __init__(self, data):
        self.data = data

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor > other.monitor

    @classmethod
    def parse_from(cls, outputs, sample_per_user: int, seperate_active=True):
        # parse from model outputs
        data = {}

        active_ratings, active_labels = [], []
        static_ratings, static_labels = [], []
        total_user_active, total_user_static = 0, 0
        all_uid, all_iid, all_rating, all_label = [], [], [], []

        for output in outputs:
            rating = output['rating']
            labels = output['labels']
            is_active = output['is_active']
            uid = output['uid']
            iid = output['iid']
            all_uid.append(uid)
            all_iid.append(iid)
            all_rating.append(rating)
            all_label.append(labels)

            if seperate_active and is_active:
                active_labels.append(labels)
                active_ratings.append(rating)
                total_user_active += labels.size(0)/sample_per_user

            else:
                static_ratings.append(rating)
                static_labels.append(labels)
                total_user_static += labels.size(0)/sample_per_user

        data = {
            'ratings': (active_ratings, static_ratings),
            'labels': (active_labels, static_labels),
            'total_user_active': total_user_active,
            'total_user_static': total_user_static,
            'total_user': total_user_active + total_user_static,
            'seperate_active': seperate_active,
            'sample_per_user': sample_per_user,
            'all_uid': all_uid,
            'all_iid': all_iid,
            'all_ratings': all_rating,
            'all_labels': all_label
        }

        print(f'total user active:{total_user_active} vs static:{total_user_static}')
        return cls(data)

    def save_for_uof(self, active_user_list, static_user_list, dataset, arch, path):
        active_df = pd.DataFrame({
            'uid': active_user_list
        })
        static_df = pd.DataFrame({
            'uid': static_user_list,
        })
        uids = torch.cat(self.data['all_uid'], dim=0).cpu().numpy()
        iids = torch.cat(self.data['all_iid'], dim=0).cpu().numpy()
        ratings = torch.cat(self.data['all_ratings'], dim=0).cpu().numpy()
        labels = torch.cat(self.data['all_labels'], dim=0).cpu().numpy()
        rank_df = pd.DataFrame({
            'uid': uids,
            'iid': iids,
            'score': ratings,
            'label': labels,
        })

        data_path = os.path.join(path, dataset, arch)
        os.makedirs(data_path, exist_ok=True)
        active_df.to_csv(os.path.join(
            data_path, "count_0.05_active_test_ratings.txt"), index=False, sep='\t')
        static_df.to_csv(os.path.join(
            data_path, "count_0.05_inactive_test_ratings.txt"), index=False, sep='\t')
        rank_df.to_csv(os.path.join(
            data_path, f"{arch}_rank.csv"), index=False, sep='\t')

    def cal_metric(self):
        # compute hit and ndcg

        hit_list_active, ndcg_list_active = [], []
        for ratings, labels in zip(self.data['ratings'][0], self.data['labels'][0]):
            hit_list_active.extend(hit_k_score_v2(
                ratings, labels, metric_K, self.data['sample_per_user']))
            ndcg_list_active.append(
                ndcg_k(labels, ratings, metric_K, self.data['sample_per_user']))

        hit_list_static, ndcg_list_static = [], []
        for ratings, labels in zip(self.data['ratings'][1], self.data['labels'][1]):
            hit_list_static.extend(hit_k_score_v2(
                ratings, labels, metric_K, self.data['sample_per_user']))
            ndcg_list_static.append(
                ndcg_k(labels, ratings, metric_K, self.data['sample_per_user']))

        hit_list_overall, ndcg_list_overall = [], []
        rating_overall = self.data['ratings'][0] + self.data['ratings'][1]
        labels_overall = self.data['labels'][0] + self.data['labels'][1]
        for ratings, labels in zip(rating_overall, labels_overall):
            hit_list_overall.extend(hit_k_score_v2(
                ratings, labels, metric_K, self.data['sample_per_user']))
            ndcg_list_overall.append(
                ndcg_k(labels, ratings, metric_K, self.data['sample_per_user']))

        self.detailed_metrics = {
            'hit_ratio@10(active)': sum(hit_list_active) / len(hit_list_active) if len(hit_list_active) > 0 else 0,
            'ndcg@10(active)': sum(ndcg_list_active) / self.data['total_user_active'] if self.data['total_user_active'] > 0 else 0,
            'hit_ratio@10(static)': sum(hit_list_static) / len(hit_list_static) if len(hit_list_static) > 0 else 0,
            'ndcg@10(static)': sum(ndcg_list_static) / self.data['total_user_static'] if self.data['total_user_static'] > 0 else 0,
            'hit_ratio@10(all)': sum(hit_list_overall) / len(hit_list_overall) if len(hit_list_overall) > 0 else 0,
            'ndcg@10(all)': sum(ndcg_list_overall) / self.data['total_user'] if self.data['total_user'] > 0 else 0
        }
        self.monitor = self.detailed_metrics['hit_ratio@10(static)']
        return self.detailed_metrics

    def save_metric(self, output_dir,  subname, dataset, seed, arch, mode, active_percent):
        performance_file_name = os.path.join(output_dir, 'performance.txt')
        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'subname': subname + active_percent + "%active",
            'dataset': dataset,
            'seed': seed,
            'metric': self.detailed_metrics,
            'architecture': arch,
            'mode': mode
        }))

    def report(self):
        for metric_name in self.detailed_metrics:
            value = self.detailed_metrics[metric_name]
            print(f'{metric_name}: {value:.4f}', end=' | ')
        print()
