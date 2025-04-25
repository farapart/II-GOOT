from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import pandas as pd
import numpy as np


def save_tsne(model,  save_dir):
    # todo get embedding
    adv_embedding, dis_embedding = model.get_embedding()
    adv_num = adv_embedding.size(0)
    dis_num = dis_embedding.size(0)
    all_embeddings = torch.cat([adv_embedding, dis_embedding], dim=0)
    x = all_embeddings.cpu().numpy()

    # applying t-sne
    tsne = TSNE(n_components=2,
                perplexity=50,
                n_iter=2000,
                init='pca',
                random_state=42,
                verbose=3)

    x_t = tsne.fit_transform(x)

    dis_ids = list(range(adv_num, adv_num + dis_num))
    adv_ids = list(range(adv_num))
    sample_dis_num = adv_num
    sample_dis_ids = np.random.choice(
        dis_ids, size=sample_dis_num, replace=True)
    sample_dis_ids = list(sample_dis_ids)
    ids = sample_dis_ids + adv_ids

    df = pd.DataFrame({
        'x': x_t[ids, 0],
        'y': x_t[ids, 1],
        "type": ['disadvantaged user'] * sample_dis_num + ["advantaged user"] * adv_num
    })
    df.to_csv(os.path.join(save_dir, f"{model.config['metric']}_tsne.csv"))

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x='x', y='y', hue='type')

    plt.savefig(os.path.join(
        save_dir, f"{model.config['metric']}_tsne.png"))


def save_tsne_1vn(model,  save_dir):
    adv_embedding, dis_embedding = model.get_embedding()
    adv_num = adv_embedding.size(0)
    dis_num = dis_embedding.size(0)
    all_embeddings = torch.cat([adv_embedding, dis_embedding], dim=0)
    x = all_embeddings.cpu().numpy()

    # applying t-sne
    tsne = TSNE(n_components=2,
                perplexity=50,
                n_iter=2000,
                init='pca',
                random_state=42,
                verbose=3)

    x_t = tsne.fit_transform(x)

    # choose by distance

    ####################
    neighbors_to_take = 2
    ####################

    adv_x = x_t[:adv_num, :]
    dis_x = x_t[adv_num:, :]
    dis_stacked = np.stack([dis_x]*adv_num, axis=0)

    print(dis_stacked.shape)
    dis_stacked = dis_stacked.reshape(
        (dis_stacked.shape[1], dis_stacked.shape[0], dis_stacked.shape[2]))
    print(dis_stacked.shape)

    distance_matrix = np.square(dis_stacked - adv_x).sum(-1)
    distance_matrix = distance_matrix.T

    sort_indexs = np.argsort(distance_matrix, axis=0)
    taken_idxs = sort_indexs[:, :neighbors_to_take].flatten()
    dis_ids = taken_idxs + adv_num

    adv_ids = list(range(adv_num))

    ids = list(dis_ids) + adv_ids

    df = pd.DataFrame({
        'x': x_t[ids, 0],
        'y': x_t[ids, 1],
        "type": ['disadvantaged user'] * len(dis_ids) + ["advantaged user"] * adv_num
    })
    df.to_csv(os.path.join(save_dir, f"{model.config['metric']}_tsne_1vn.csv"))

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x='x', y='y', hue='type')

    plt.savefig(os.path.join(
        save_dir, f"{model.config['metric']}_tsne_1v{neighbors_to_take}.png"))
