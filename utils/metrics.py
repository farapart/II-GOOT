import torch
from sklearn.metrics import ndcg_score

def hit_k_score_v2(predict: torch.FloatTensor, labels: torch.LongTensor, k: int, sample_per_user: int) -> list:
    predict = torch.reshape(predict, (-1, sample_per_user))
    labels = torch.reshape(labels, (-1, sample_per_user))
    values, indices = torch.topk(predict, k, dim=-1)

    assert values.size(0) == predict.size(0)
    hit_list = []
    indices_list = indices
    for label, indices in zip(labels, indices_list):
        topk_labels = label[indices]
        hit_list.append((torch.sum(topk_labels) > 0).item())
    return hit_list


def ndcg_k(labels: torch.IntTensor, predict: torch.FloatTensor, k: int, sample_per_user: int) -> list:
    labels = labels.cpu().numpy()
    labels = labels.reshape(-1, sample_per_user)
    predict = predict.cpu().numpy()
    predict = predict.reshape(-1, sample_per_user)

    return ndcg_score(labels, predict, k=k) * labels.shape[0]



