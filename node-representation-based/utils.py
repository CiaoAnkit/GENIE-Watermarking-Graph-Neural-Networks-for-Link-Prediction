import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def test_wm(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    pos_test_edge = split_edge['test']['edge'].to(h.device).to(torch.long)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device).to(torch.long)
    pos_wm_edge = split_edge['watermark']['edge'].to(h.device).to(torch.long)
    neg_wm_edge = split_edge['watermark']['edge_neg'].to(h.device).to(torch.long)

    h = model(data.x, data.full_adj_t)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    pos_wm_preds = []
    for perm in DataLoader(range(pos_wm_edge.size(0)), 128):
        edge = pos_wm_edge[perm].t()
        pos_wm_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_wm_pred = torch.cat(pos_wm_preds, dim=0)
    # print(pos_wm_pred)

    neg_wm_preds = []
    for perm in DataLoader(range(neg_wm_edge.size(0)), 128):
        edge = neg_wm_edge[perm].t()
        neg_wm_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_wm_pred = torch.cat(neg_wm_preds, dim=0)
    # print(neg_wm_pred)

    test_pred = torch.cat((pos_test_pred,neg_test_pred),dim=0)
    ones_tensor = torch.ones(pos_test_pred.shape[0])
    zeros_tensor = torch.zeros(neg_test_pred.shape[0])
    test_true = torch.cat((ones_tensor, zeros_tensor), dim=0)

    wm_pred = torch.cat((pos_wm_pred,neg_wm_pred),dim=0)
    ones_tensor = torch.ones(pos_wm_pred.shape[0])
    zeros_tensor = torch.zeros(neg_wm_pred.shape[0])
    wm_true = torch.cat((ones_tensor, zeros_tensor), dim=0)

    auc_score_test = roc_auc_score(test_true,test_pred)
    auc_score_wm = roc_auc_score(wm_true,wm_pred)

    return auc_score_test,auc_score_wm

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    positive_test_edge = split_edge['test']['edge'].to(h.device).to(torch.long)
    negative_test_edge = split_edge['test']['edge_neg'].to(h.device).to(torch.long)

    h = model(data.x, data.full_adj_t)

    positive_test_preds = []
    for perm in DataLoader(range(positive_test_edge.size(0)), batch_size):
        edge = positive_test_edge[perm].t()
        positive_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(positive_test_preds, dim=0)

    negative_test_preds = []
    for perm in DataLoader(range(negative_test_edge.size(0)), batch_size):
        edge = negative_test_edge[perm].t()
        negative_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(negative_test_preds, dim=0)

    test_pred = torch.cat((pos_test_pred,neg_test_pred),dim=0)
    ones_tensor = torch.ones(pos_test_pred.shape[0])
    zeros_tensor = torch.zeros(neg_test_pred.shape[0])
    test_true = torch.cat((ones_tensor, zeros_tensor), dim=0)

    auc_score_test = roc_auc_score(test_true,test_pred)

    return auc_score_test