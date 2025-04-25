import torch
import torch.nn as nn
from .model_configs import mf_config as config
import numpy as np
import ot


class MF(torch.nn.Module):
    def __init__(self, user_num, item_num, active_user_list: list, static_user_list: list,
                 dis_half2half: dict, mode: str,
                 static2active_v2: dict):
        super(MF, self).__init__()

        # fairness related attributes
        self.active_user_list = torch.tensor(active_user_list, dtype=torch.long).cuda()
        self.static_user_list = torch.tensor(static_user_list, dtype=torch.long).cuda()
        self.static_user_set = set(static_user_list)
        self.active_num = len(active_user_list)
        self.static_num = len(static_user_list)
        self.static2active_v2 = static2active_v2
        self.dis_half2half = dis_half2half

        self.mode = mode

        self.num_users = user_num
        self.num_items = item_num
        self.latent_dim = config['latent_dim_mf']
        self.config = config
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)
        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, user_indices, item_indices, labels, is_active=False):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        if self.mode in ('intra', ) and not is_active:
            corresponding_users = torch.LongTensor([self.dis_half2half[user] for user in list(user_indices.cpu().numpy())]).cuda()

            combined_user_embeddings = torch.cat([user_embedding, self.embedding_user(corresponding_users)], dim=0)
            combined_item_embeddings = torch.cat([item_embedding, item_embedding], dim=0)

            predict_rating = self.sigmoid(torch.sum(torch.mul(combined_user_embeddings, combined_item_embeddings), dim=1))
            ori_rating, corres_rating = predict_rating.chunk(2)

            ori_loss = self.loss(ori_rating, labels)
            ot_loss = self.loss(corres_rating, labels)

            return {
                'loss': ori_loss + ot_loss,
                'original_loss': ori_loss,
                'OT_loss_intra': ot_loss,
                'OT_loss_inter': torch.tensor(0),
                'rating': ori_rating,
                'labels': labels
            }
        elif self.mode in ('inter_v3', ) and not is_active:

            predict_rating = self.sigmoid(torch.sum(torch.mul(user_embedding, item_embedding), dim=1))
            ori_loss = self.loss(predict_rating, labels)

            # inter_v3
            corres_adv = []
            for dis_u in user_indices:
                corres_adv.append(self.static2active_v2[dis_u.item()])
            corres_indice = torch.tensor(corres_adv, dtype=torch.long).cuda()
            corres_embedding = self.embedding_user(corres_indice)
            ot_loss_inter = torch.mean(torch.sum(torch.square(corres_embedding - user_embedding), dim=1))
            return {
                'loss': ori_loss + ot_loss_inter,
                'original_loss': ori_loss,
                'OT_loss_intra': torch.tensor(0),
                'OT_loss_inter': ot_loss_inter,
                'rating': predict_rating,
                'labels': labels
            }
        elif self.mode in ('fair_v3', ) and not is_active:
            # intra
            corresponding_users = torch.LongTensor([self.dis_half2half[user] for user in list(user_indices.cpu().numpy())]).cuda()
            combined_user_embeddings = torch.cat([user_embedding, self.embedding_user(corresponding_users)], dim=0)
            combined_item_embeddings = torch.cat([item_embedding, item_embedding], dim=0)

            predict_rating = self.sigmoid(torch.sum(torch.mul(combined_user_embeddings, combined_item_embeddings), dim=1))
            ori_rating, corres_rating = predict_rating.chunk(2)

            ori_loss = self.loss(ori_rating, labels)
            ot_loss_intra = self.loss(corres_rating, labels)

            # inter_v3
            corres_adv = []
            for dis_u in user_indices:
                corres_adv.append(self.static2active_v2[dis_u.item()])
            corres_indice = torch.tensor(corres_adv, dtype=torch.long).cuda()
            corres_embedding = self.embedding_user(corres_indice)
            ot_loss_inter = torch.mean(
                torch.sum(torch.square(corres_embedding - user_embedding), dim=1))
            return {
                'loss': ori_loss,
                'original_loss': ori_loss,
                'OT_loss_intra': ot_loss_intra,
                'OT_loss_inter': ot_loss_inter,
                'rating': ori_rating,
                'labels': labels
            }
        else:
            # normal or a batch of advantaged users
            predict_rating = self.sigmoid(
                torch.sum(torch.mul(user_embedding, item_embedding), dim=1))
            loss = self.loss(predict_rating, labels)
            return {
                'loss': loss,
                'original_loss': loss,
                'OT_loss_inter': torch.tensor(0),
                'OT_loss_intra': torch.tensor(0),
                'rating': predict_rating,
                'labels': labels
            }

