import torch
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np


def split_data(
    array_of_edges: np.array,
    split_ratio: list,
    seed: int,
    dataset_shuffle: bool,
) -> dict:
    """Split your dataset into train, valid, and test

    Args:
        array_of_edges (np.array): array_of_edges
        split_ratio (list): train:test:val = [float, float, float], train+test+val = 1.0 
        seed (int) = seed
        dataset_shuffle (bool) = shuffle dataset when split

    Returns:
        dataset_dict: {train_edges : np.array, train_label : np.array, test_edges: np.array, test_labels: np.array, valid_edges: np.array, valid_labels: np.array}
    """

    assert np.isclose(sum(split_ratio), 1), "train+test+valid != 1"
    train_ratio, valid_ratio, test_ratio = split_ratio
    train_X, test_val_X, train_Y, test_val_Y = train_test_split(
        array_of_edges[:, :2], array_of_edges[:, 2], test_size=1 - train_ratio, random_state=seed, shuffle=dataset_shuffle)
    val_X, test_X, val_Y, test_Y = train_test_split(test_val_X, test_val_Y, test_size=test_ratio/(
        test_ratio + valid_ratio), random_state=seed, shuffle=dataset_shuffle)

    dataset_dict = {
        "train_edges": train_X,
        "train_label": train_Y,
        "valid_edges": val_X,
        "valid_label": val_Y,
        "test_edges": test_X,
        "test_label": test_Y
    }

    return dataset_dict


def load_data(
    dataset_path: str,
    direction: bool,
) -> np.array:
    """Read data from a file

    Args:
        dataset_path (str): dataset_path
        direction (bool): True=direct, False=undirect
        node_idx_type (str): "uni" - no intersection with [uid, iid], "bi" - [uid, iid] idx has intersection

    Return:
        array_of_edges (array): np.array of edges
        num_of_nodes: [type1(int), type2(int)]
    """

    edgelist = []
    with open(dataset_path) as f:
        for line in f:
            a, b, s = map(int, line.split('\t'))
            if s == -1:
                s = 0
            edgelist.append((a, b, s))
    num_of_nodes = get_num_nodes(np.array(edgelist))
    edgelist = np.array(edgelist)

    assert max(edgelist[:,0]) > min(edgelist[:,1]), "user id and item id must be separated"
    edgelist[:,1] = edgelist[:,1] + num_of_nodes[0]
    if direction == False:
        edgelist_2 = edgelist.tolist()
        temp_edgelist = []
        for idx, edge in enumerate(edgelist):
            fr, to, sign = edge
            temp_edgelist.append([to, fr, sign])
        edgelist_temp = np.array(temp_edgelist)
        edgelist = np.concatenate((edgelist, edgelist_temp), axis=0)
    num_edges = np.array(edgelist).shape[0]
    
    return edgelist, num_of_nodes, num_edges


def get_num_nodes(
    dataset: np.array
) -> int:
    """get num nodes when bipartite

    Args:
        dataset (np.array): dataset

    Returns:
        num_nodes tuple(int, int): num_nodes_user, num_nodes_item
    """
    num_nodes_user = np.amax(dataset[:, 0]) + 1 #idx max is 6039
    num_nodes_item = np.amax(dataset[:, 1]) + 1 #idx max is 3951
    return (num_nodes_user.item(), num_nodes_item.item())


def collate_fn(batch):
    user, items = zip(*batch)
    return user, items

def edgelist_to_user_item_dict(edgelist: np.array, direction, num_nodes) -> dict:
    user_item_dict = defaultdict(list)    
    for edge in edgelist:
        if direction:
            user_item_dict[edge[0]].append(edge[1])
        else:
            if edge[0] < num_nodes[0] and edge[1] >= num_nodes[0]:
                user_item_dict[edge[0]].append(edge[1])
    return user_item_dict

def rwr(edgelist, num_nodes, iter_K, alpha, device):
    sum_nodes = sum(num_nodes)
    A = torch.sparse_coo_tensor(torch.tensor(edgelist).T, torch.tensor([1]*edgelist.shape[0]), torch.Size([sum_nodes, sum_nodes]), dtype=torch.float32)
    A = torch.eye(sum_nodes) + A
    row_sum = torch.sum(A, dim=1)
    d_inv_row = 1.0 / row_sum.to_dense()
    d_inv_row[torch.isinf(d_inv_row)] = 0
    d_inv_matrix = torch.diag(d_inv_row)
    A = A.to(torch.float32)
    nA = torch.sparse.mm(d_inv_matrix, A)
    nAT = nA.T.to(device)
    
    x0 = torch.eye(sum_nodes).to(device)
    I = torch.eye(sum_nodes).to(device)
    x = x0
    for i in range(iter_K):
        x = (1-alpha) * torch.sparse.mm(nAT, x) + alpha * I

    return x.T

def rwr_with_filter(edgelist, num_nodes, iter_K, alpha, device, eps):
    sum_nodes = sum(num_nodes)
    A = torch.sparse_coo_tensor(torch.tensor(edgelist).T, torch.tensor([1]*edgelist.shape[0]), torch.Size([sum_nodes, sum_nodes]), dtype=torch.float32)
    
    A = torch.eye(sum_nodes) + A
    row_sum = torch.sum(A, dim=1)
    d_inv_row = 1.0 / row_sum.to_dense()
    d_inv_row[torch.isinf(d_inv_row)] = 0
    d_inv_matrix = torch.diag(d_inv_row)
    A = A.to(torch.float32)
    nA = torch.sparse.mm(d_inv_matrix, A)
    nAT = nA.T.to(device)
    
    x0 = torch.eye(sum_nodes).to(device)
    I = torch.eye(sum_nodes).to(device)
    x = x0
    for i in range(iter_K):
        x = (1-alpha) * torch.sparse.mm(nAT, x) + alpha * I
    x = torch.where(x < eps, torch.tensor(0.0).to(device), x)
    
    
    if torch.sum(torch.sum(x, dim=0) == torch.sum(x, dim=1)) == x.shape[0]:
        raise ValueError("filtering eps is too high, it remains only diag elements")
    
    x = x.T
    
    eye_matrix = torch.eye(x.size(0)).to(device)
    x = x * (1 - eye_matrix)
    
    row_sum = torch.sum(x.abs(), dim=1).float() #row sum
    d_inv_row = torch.pow(row_sum, -0.5).flatten()
    d_inv_row[torch.isinf(d_inv_row)] = 0.
    d_mat_row = torch.diag(d_inv_row)
    norm_adj = d_mat_row @ x.to_dense()  
    
    return norm_adj
    
def rwr_with_non_weighted_diffusion(edgelist, num_nodes, iter_K, alpha, device, eps):
    sum_nodes = sum(num_nodes)
    A = torch.sparse_coo_tensor(torch.tensor(edgelist).T, torch.tensor([1]*edgelist.shape[0]), torch.Size([sum_nodes, sum_nodes]), dtype=torch.float32)
    
    A = torch.eye(sum_nodes) + A
    row_sum = torch.sum(A, dim=1)
    d_inv_row = 1.0 / row_sum.to_dense()
    d_inv_row[torch.isinf(d_inv_row)] = 0
    d_inv_matrix = torch.diag(d_inv_row)
    A = A.to(torch.float32)
    nA = torch.sparse.mm(d_inv_matrix, A)
    nAT = nA.T.to(device)
    
    x0 = torch.eye(sum_nodes).to(device)
    I = torch.eye(sum_nodes).to(device)
    x = x0
    for i in range(iter_K):
        x = (1-alpha) * torch.sparse.mm(nAT, x) + alpha * I
    file = open("tradeoff.txt", "a")
    file.write(f"eps: {eps}, alpha: {alpha}, before_filtered: {sum(sum(torch.where(x > 0, torch.tensor(1.0).to(device), x)))}  ")
    breakpoint()
    x = torch.where(x < eps, torch.tensor(0.0).to(device), x)
    
    
    if torch.sum(torch.sum(x, dim=0) == torch.sum(x, dim=1)) == x.shape[0]:
        file.write(f"after_filtering: remain only diag elements \n")
        file.close()
        raise ValueError("filtering eps is too high, it remains only diag elements")
    
    x = x.T
    x = torch.where(x > 0, torch.tensor(1.0).to(device), x)
    
    #Remove Diag elements
    eye_matrix = torch.eye(x.size(0)).to(device)
    x = x * (1 - eye_matrix)
    
    
    file.write(f"after_filtering: {sum(sum(x))}\n")
    file.close()
    row_sum = torch.sum(x.abs(), dim=1).float() #row sum
    d_inv_row = torch.pow(row_sum, -0.5).flatten()
    d_inv_row[torch.isinf(d_inv_row)] = 0.
    d_mat_row = torch.diag(d_inv_row)
    norm_adj = d_mat_row @ x.to_dense()  
    
    return norm_adj