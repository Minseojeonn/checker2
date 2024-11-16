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
    sign: int
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
            a, b, s, d = map(int, line.split('::'))
            if sign != -1:
                if s >= sign:
                    s = 1
                else:
                    s = -1
            edgelist.append((a, b, s))
    num_of_nodes = get_num_nodes(np.array(edgelist))
    edgelist = np.array(edgelist)
    assert max(edgelist[:,0]) > min(edgelist[:,1]), "user id and item id must be separated"
    edgelist[:,1] = edgelist[:,1] + num_of_nodes[0]
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
        user_item_dict[edge[0]].append(edge[1])
        
    return user_item_dict
    