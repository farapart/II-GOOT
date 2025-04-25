import ot
import torch
import numpy as np
from .model_configs import neumf_config as config
import torch.nn as nn


class NeuMF(torch.nn.Module):
    def __init__(self, user_num, item_num, active_user_list: list, static_user_list: list,
                 #  static2active: dict, half2half: dict, active2static:dict,
                 dis_half2half: dict, mode: str,
                 static2active_v2: dict):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = user_num
        self.num_items = item_num

        # fairness related attributes
        self.active_user_list = torch.tensor(active_user_list, dtype=torch.long).cuda()
        self.static_user_list = torch.tensor(static_user_list, dtype=torch.long).cuda()
        self.static_user_set = set(static_user_list)
        self.active_num = len(active_user_list)
        self.static_num = len(static_user_list)
        self.dis_half2half = dis_half2half
        self.static2active_v2 = static2active_v2

        self.mode = mode

        # construct model
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, user_indices, item_indices, labels, is_active=False):
        '''

        Args:
            user_indices: [B, ]
            item_indices: [B, ]
            labels: [B, ] ,interaction labels, 1 if interaction exist else 0
            is_active: whether the users in a batch is advantaged or not

        Returns:
            dict:
                loss: bce_loss according to NCF paper
                rating: predicted rating of each interaction, values in [0,1]
                labels: [B, ] ,interaction labels, 1 if interaction exist else 0
                loss: overall loss
                original_loss: loss computed using entropy of user predictions
                OT_loss_inter: inter group stage, the distance between disadvantaged
                               user and their precomputed corresponding advantaged user
                OT_loss_intra: intra group stage, adding the interaction to enhance the 
                               training of corresponding disadvantaged users
        '''

        if self.mode in ("intra", ) and not is_active:
            # get the most similar user pair for each user
            corresponding_users = torch.LongTensor([self.dis_half2half[user] for user in list(user_indices.cpu().numpy())]).cuda()

            all_user_indices = torch.cat([user_indices, corresponding_users], dim=0)
            all_item_indices = torch.cat([item_indices, item_indices], dim=0)

            user_embedding_mlp = self.embedding_user_mlp(all_user_indices)
            item_embedding_mlp = self.embedding_item_mlp(all_item_indices)
            user_embedding_mf = self.embedding_user_mf(all_user_indices)
            item_embedding_mf = self.embedding_item_mf(all_item_indices)

            # the concat latent vector
            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)
            logits = self.affine_output(vector)
            logits = torch.squeeze(logits)
            ori_logits, corres_logits = logits.chunk(2)

            rating = self.logistic(logits)
            ori_rating, corres_rating = rating.chunk(2)

            ori_loss = self.loss(ori_logits, labels)
            ot_loss = self.loss(corres_logits, labels)

            return {
                'loss': ori_loss + ot_loss,
                'original_loss': ori_loss,
                'OT_loss_intra': ot_loss,
                'OT_loss_inter': torch.tensor(0),
                'rating': ori_rating,
                'labels': labels
            }
        elif self.mode in ("inter", ) and not is_active:
            # normal 
            user_embedding_mlp = self.embedding_user_mlp(user_indices)
            item_embedding_mlp = self.embedding_item_mlp(item_indices)
            user_embedding_mf = self.embedding_user_mf(user_indices)
            item_embedding_mf = self.embedding_item_mf(item_indices)

            # the concat latent vector
            mlp_vector = torch.cat(
                [user_embedding_mlp, item_embedding_mlp], dim=-1)
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)
            logits = self.affine_output(vector)
            logits = torch.squeeze(logits)
            rating = self.logistic(logits)
            original_loss = self.loss(logits, labels)

            # inter
            corres_adv = []
            for dis_u in user_indices:
                corres_adv.append(self.static2active_v2[dis_u.item()])
            corres_indice = torch.tensor(corres_adv, dtype=torch.long).cuda()
            corres_embedding = torch.cat([self.embedding_user_mlp(corres_indice), self.embedding_user_mf(corres_indice)], dim=-1)
            cat_disadvantaged_embedding = torch.cat([user_embedding_mlp, user_embedding_mf], dim=-1)
            ot_loss_inter = torch.mean(
                torch.sum(torch.square(corres_embedding - cat_disadvantaged_embedding), dim=1))
            return {
                'loss': original_loss + ot_loss_inter,
                'original_loss': original_loss,
                'OT_loss_intra': torch.tensor(0),
                'OT_loss_inter': ot_loss_inter,
                'rating': rating,
                'labels': labels
            }
        elif self.mode in ("fair", ) and not is_active:
            # normal and intra
            corresponding_users = torch.LongTensor(
                [self.dis_half2half[user] for user in list(user_indices.cpu().numpy())]).cuda()

            all_user_indices = torch.cat(
                [user_indices, corresponding_users], dim=0)
            all_item_indices = torch.cat([item_indices, item_indices], dim=0)

            user_embedding_mlp = self.embedding_user_mlp(all_user_indices)
            item_embedding_mlp = self.embedding_item_mlp(all_item_indices)
            user_embedding_mf = self.embedding_user_mf(all_user_indices)
            item_embedding_mf = self.embedding_item_mf(all_item_indices)

            # the concat latent vector
            mlp_vector = torch.cat(
                [user_embedding_mlp, item_embedding_mlp], dim=-1)
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)
            logits = self.affine_output(vector)
            logits = torch.squeeze(logits)
            ori_logits, corres_logits = logits.chunk(2)

            rating = self.logistic(logits)
            ori_rating, corres_rating = rating.chunk(2)

            ori_loss = self.loss(ori_logits, labels)
            ot_loss_intra = self.loss(corres_logits, labels)

            # inter
            corres_adv = []
            for dis_u in user_indices:
                corres_adv.append(self.static2active_v2[dis_u.item()])
            corres_indice = torch.tensor(corres_adv, dtype=torch.long).cuda()
            corres_embedding = torch.cat([self.embedding_user_mlp(corres_indice), self.embedding_user_mf(corres_indice)], dim=-1)
            dis_embedding_mf, _ = torch.chunk(user_embedding_mf, 2, dim=0)
            dis_embedding_mlp, _ = torch.chunk(user_embedding_mlp, 2, dim=0)
            cat_disadvantaged_embedding = torch.cat(
                [dis_embedding_mlp, dis_embedding_mf], dim=-1)
            ot_loss_inter = torch.mean(torch.sum(torch.square(corres_embedding - cat_disadvantaged_embedding), dim=1))
            return {
                'loss': ori_loss,
                'original_loss': ori_loss,
                'OT_loss_intra': ot_loss_intra,
                'OT_loss_inter': ot_loss_inter,
                'rating': ori_rating,
                'labels': labels
            }
        else:
            # normal
            user_embedding_mlp = self.embedding_user_mlp(user_indices)
            item_embedding_mlp = self.embedding_item_mlp(item_indices)
            user_embedding_mf = self.embedding_user_mf(user_indices)
            item_embedding_mf = self.embedding_item_mf(item_indices)

            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)
            logits = self.affine_output(vector)
            logits = torch.squeeze(logits)
            rating = self.logistic(logits)
            original_loss = self.loss(logits, labels)
            return {
                'loss': original_loss,
                'original_loss': original_loss,
                'OT_loss_intra': torch.tensor(0),
                'OT_loss_inter': torch.tensor(0),
                'rating': rating,
                'labels': labels
            }

    