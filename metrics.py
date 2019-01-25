import pickle
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pairwise_dist(x, y):
    '''
    Args: x is a Nxd matrix
          y is an optional Mxd matirx
    Return: dist is a NxM matrix where dist[i,j] is ||x[i,:]-y[j,:]||^2
    from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def laplace_loss(pred, pred_b, graph, i):
    '''
    Compute the laplacian loss
    '''
    # Set last elem to zero (default value for padding)
    ext_pred = torch.cat((pred, torch.zeros(1, 3)))
    ext_pred_b = torch.cat((pred_b, torch.zeros(1, 3)))
    # Compute Laplacian coordinates
    lap = pred - torch.sum(ext_pred[graph.maps[i//2]], dim=1) / graph.len_maps[i//2]
    lap_b = pred_b - torch.sum(ext_pred_b[graph.maps[i//2]], dim=1) / graph.len_maps[i//2]
    return torch.mean(torch.norm(lap - lap_b, dim = 1)**2)

def normal_loss(normals, min_idx, diff, graph, i):
    '''
    Compute the normal loss
    '''
    return torch.mean(torch.sum(torch.mul(
                normals[min_idx[graph.adjacency_mat[i//2][0]]], diff), dim=1)**2)

def edge_loss(pred, graph, i):
    '''
    Compute the edge loss
    '''
    v_n = pred[graph.adjacency_mat[i//2]]
    diff = v_n[0, :, :] - v_n[1, :, :]
    return diff, torch.mean(torch.norm(diff, dim=1)**2)

def chamfer_loss(pred, gt, normalized=True):
    '''
    Compute the chamfer loss
    '''
    dist = pairwise_dist(pred, gt)
    min_dist, min_idx = dist.min(dim=1)
    if normalized:
        return min_idx, torch.mean(min_dist) + torch.mean(dist.min(dim=0)[0])
    else:
        return min_idx, torch.sum(min_dist) + torch.sum(dist.min(dim=0)[0])

def loss_function(preds, gt_points, gt_normals, graph):
    '''
    Compute the total loss for the three meshes
    Args:
        preds: series of predicted meshes from the ellipsoid (preds[0]) to final
               mesh (preds[5]). Contains also intermediate meshes (before unpooling)
               for computing laplacian loss
        gt_points: Ground truth mesh
        gt_normals: Ground truth normals
        graph: Graph data structure defined in graph.py
    '''
    total_loss = 0
    for i in range(1, len(preds), 2):
        pred = preds[i]
        pred_b = preds[i-1] # before unpooling

        # Chamfer Loss
        min_idx, chamf_loss = chamfer_loss(pred, gt_points)
        # Edge loss
        diff, ed_loss = edge_loss(pred, graph, i)
        # Laplace loss
        lap_loss = laplace_loss(pred, pred_b, graph, i)
        # Normal loss
        norm_loss = normal_loss(gt_normals, min_idx, diff, graph, i)

        # Weighted loss for current mesh
        total_loss += chamf_loss + 0.1 * ed_loss + 0.3 * lap_loss + 1e-4 * norm_loss

    return total_loss

def f1_score(pred, gt, threshold):
    '''
    Compute a F1 score based on a precision and a recall score computed from
    the number of points in prediction or ground truth that can find a
    nearest neighbor from the other within certain threshold
    '''
    dist = pairwise_dist(pred, gt)
    min_dist_1, _ = dist.min(dim=0)
    min_dist_2, _ = dist.min(dim=1)
    precision = min_dist_2[min_dist_2 < threshold].size(0)/min_dist_2.size(0)
    recall = min_dist_1[min_dist_1 < threshold].size(0)/min_dist_1.size(0)
    return 2 * (precision * recall)/(precision + recall + 1e-8)


if __name__ == "__main__":
    print("Testing loss")
    v = torch.Tensor([[0, 0, 1], [0, 2, 1], [3, 1, 0], [1, 1, 1]])
    vb = torch.Tensor([[0, 1, 1], [0, 2, 1], [3, 1, 0], [1, 1, 1]])
    e = torch.Tensor([[1, 1, 2, 3, 3, 4], [3, 2, 1, 1, 4, 3]]).long()
    n = torch.Tensor([[1, 2, 0], [3, -1, 2], [-0.5, 4, -6], [1, -1, 1]])

    from graph import Graph
    graph = Graph("./ellipsoid/init_info.pickle")
    map = torch.Tensor([[2, 1], [0, 0], [0, 3], [2, 0]]).long()
    graph.maps[0] = map
    graph.len_maps[0] = torch.Tensor([2, 1, 2, 1]).view(-1, 1)
    graph.adjacency_mat[0]= e
    graph.adjacency_mat[0][0] -= 1
    v_n = v[e-1]
    p_minus_k = v_n[0, :, :] - v_n[1, :, :]
    edge_loss = torch.mean(torch.norm(p_minus_k, dim=1)**2)
    min_idx, chamf_loss = chamfer_loss(v, vb)
    norm_loss = normal_loss(n, min_idx, p_minus_k, graph, 0)

    print("F1 Score", f1_score(v, vb, threshold=1e-4))
    print("Chamfer Loss", chamf_loss)
    print("Laplacian loss", laplace_loss(v, vb, graph, 0))
    print("Edge loss", edge_loss)
    print("Normal loss", norm_loss)
