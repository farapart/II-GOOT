import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot
import dgl
import dgl.function as fn
from .model_configs import LightGCN_config as config


class LightGCNLayer(nn.Module):
    def __init__(self, g, norm_dict, dropout):
        super(LightGCNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            norm = self.norm_dict[(srctype, etype, dsttype)]
            messages = norm * feat_dict[srctype][src]
            g.edges[(srctype, etype, dsttype)].data[etype] = messages
            funcs[(srctype, etype, dsttype)] = (
                fn.copy_e(etype, 'm'), fn.sum('m', 'h'))

        g.multi_update_all(funcs, 'sum')
        feature_dict = {}
        for ntype in g.ntypes:
            h = g.nodes[ntype].data['h']
            feature_dict[ntype] = h
        return feature_dict


class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, active_user_list: list, static_user_list: list,
                 dis_half2half: dict, mode: str,
                 inter_user_list: list, inter_item_list: list,
                 static2active_v2: dict):
        super(LightGCN, self).__init__()

        # fairness related attributes
        self.active_num = len(active_user_list)
        self.static_num = len(static_user_list)
        self.active_user_list = torch.tensor(active_user_list, dtype=torch.long).cuda()
        self.static_user_list = torch.tensor(static_user_list, dtype=torch.long).cuda()
        self.dis_half2half = dis_half2half
        self.static2active_v2 = static2active_v2
        self.mode = mode

        self.norm_dict = dict()
        self.config = config

        # build graph
        self.g = self.build_graph(
            inter_user_list, inter_item_list, user_num, item_num).to("cuda")
        for srctype, etype, dsttype in self.g.canonical_etypes:
            src, dst = self.g.edges(etype=(srctype, etype, dsttype))
            dst_degree = self.g.in_degrees(dst, etype=(srctype, etype, dsttype)).float()  # obtain degrees
            src_degree = self.g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1)  # compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm

        # contruct lightgcn model
        self.layers = nn.ModuleList()
        for i in range(len(config['layer_size'])):
            self.layers.append(
                LightGCNLayer(self.g, self.norm_dict, config['dropout'][0])
            )

        # construct initializer 
        self.initializer = nn.init.normal_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(self.g.num_nodes(ntype), config['in_size']), std=0.01)) for ntype in self.g.ntypes
            # ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, user_indices, item_indices, labels, is_active=False, user_key='user', item_key='item'):
        h_dict = {ntype: self.feature_dict[ntype] for ntype in self.g.ntypes}

        # get user and item embeddings
        user_embed = h_dict[user_key]
        item_embed = h_dict[item_key]
        for i, layer in enumerate(self.layers):
            h_dict = layer(self.g, h_dict)
            user_embed = user_embed + h_dict[user_key]*(1/(i+2))
            item_embed = item_embed + h_dict[item_key]*(1/(i+2))

        # compute prediction and fairness related loss
        if self.mode in ('inter', "normal") and not is_active:
            user_embeddings = user_embed[user_indices, :]
            item_embeddings = item_embed[item_indices, :]
            pred_rating = self.sigmoid(
                torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1))
            loss = self.loss(pred_rating, labels)

            # inter v3
            corres_adv = []
            for dis_u in user_indices:
                corres_adv.append(self.static2active_v2[dis_u.item()])
            corres_indice = torch.tensor(corres_adv, dtype=torch.long)
            corres_embedding = user_embed[corres_indice, :]
            ot_loss = torch.mean(torch.sum(torch.square(
                corres_embedding - user_embeddings), dim=1))
            return {
                'loss': loss,
                'original_loss': loss,
                'OT_loss_intra': torch.tensor(0),
                'OT_loss_inter': ot_loss,
                'rating': pred_rating,
                'labels': labels
            }
        elif self.mode in ('fair',) and not is_active:
            # intra
            corresponding_users = torch.LongTensor([self.dis_half2half[user] for user in list(user_indices.cpu().numpy())]).cuda()
            corres_user_embd = user_embed[corresponding_users, :]
            ori_user_embd = user_embed[user_indices, :]
            item_embeddings = item_embed[item_indices, :]

            combined_user_embd = torch.cat([ori_user_embd, corres_user_embd], dim=0)
            combined_item_embd = torch.cat([item_embeddings, item_embeddings], dim=0)

            predict_rating = self.sigmoid(torch.sum(torch.mul(combined_user_embd, combined_item_embd), dim=1))
            ori_rating, corres_rating = predict_rating.chunk(2)

            ori_loss = self.loss(ori_rating, labels)
            ot_loss_intra = self.loss(corres_rating, labels)

            # inver_v3
            corres_adv = []
            for dis_u in user_indices:
                corres_adv.append(self.static2active_v2[dis_u.item()])
            corres_indice = torch.tensor(corres_adv, dtype=torch.long)
            corres_embedding = user_embed[corres_indice, :]
            ot_loss_inter = torch.mean(
                torch.sum(torch.square(corres_embedding - ori_user_embd), dim=1))
            return {
                'loss': ori_loss,
                'original_loss': ori_loss,
                'OT_loss_intra': ot_loss_intra,
                'OT_loss_inter': ot_loss_inter,
                'rating': ori_rating,
                'labels': labels
            }
        elif self.mode in ("intra", ) and not is_active:
            # intra
            corresponding_users = torch.LongTensor(
                [self.dis_half2half[user] for user in list(user_indices.cpu().numpy())]).cuda()
            corres_user_embd = user_embed[corresponding_users, :]
            ori_user_embd = user_embed[user_indices, :]
            item_embeddings = item_embed[item_indices, :]

            combined_user_embd = torch.cat(
                [ori_user_embd, corres_user_embd], dim=0)
            combined_item_embd = torch.cat(
                [item_embeddings, item_embeddings], dim=0)

            predict_rating = self.sigmoid(
                torch.sum(torch.mul(combined_user_embd, combined_item_embd), dim=1))
            ori_rating, corres_rating = predict_rating.chunk(2)

            ori_loss = self.loss(ori_rating, labels)
            ot_loss_intra = self.loss(corres_rating, labels)
            return {
                'loss': ori_loss + ot_loss_intra,
                'original_loss': ori_loss,
                'OT_loss_intra': ot_loss_intra,
                'OT_loss_inter': torch.tensor(0),
                'rating': ori_rating,
                'labels': labels
            }
        else:
            # normal or the batch does not contains advantaged users
            user_embeddings = user_embed[user_indices, :]
            item_embeddings = item_embed[item_indices, :]
            pred_rating = self.sigmoid(torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1))
            loss = self.loss(pred_rating, labels)

            return {
                'loss': loss,
                'original_loss': loss,
                'OT_loss_intra': torch.tensor(0),
                'OT_loss_inter': torch.tensor(0),
                'rating': pred_rating,
                'labels': labels
            }

    def build_graph(self, train_users, train_items, num_users, num_items):
        user_selfs = [i for i in range(num_users)]
        item_selfs = [i for i in range(num_items)]

        data_dict = {
            ('user', 'user_self', 'user'): (user_selfs, user_selfs),
            ('item', 'item_self', 'item'): (item_selfs, item_selfs),
            ('user', 'ui', 'item'): (train_users, train_items),
            ('item', 'iu', 'user'): (train_items, train_users)
        }

        num_dict = {
            'user': num_users, 'item': num_items
        }

        g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
        return g


    def get_embedding(self, user_key='user'):
        h_dict = {ntype: self.feature_dict[ntype] for ntype in self.g.ntypes}
        user_embeds = []
        user_embeds.append(h_dict[user_key])
        for layer in self.layers:
            h_dict = layer(self.g, h_dict)
            user_embeds.append(h_dict[user_key])
        user_embd = torch.cat(user_embeds, 1)

        active_user_embedding = user_embd[self.active_user_list, :]
        static_user_embedding = user_embd[self.static_user_list, :]
        return active_user_embedding, static_user_embedding
