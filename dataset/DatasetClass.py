import torch
import numpy as np
import torch.utils.data as data

from dataset.utils import edgelist_to_user_item_dict

class TrnDatasetClass(data.Dataset):
    """
    Dataset class with optimized negative sampling using numpy for set operations.
    
    args:
        edge (np.array): edge
        label (np.array): label
        user_item_dict (dict): dictionary with user-item interactions
        num_nodes (tuple): (num_users, num_items)
    
    return:
        Dataset ready for DataLoader
    """
    def __init__(self, direction, edge, label, num_nodes: tuple, device='cpu') -> None:
        self.device = device
        self.num_nodes = num_nodes
        self.user_item_dict = edgelist_to_user_item_dict(edge, direction, num_nodes)
        # Negative sampling optimized to run during initialization
        self.unseen_item_dict = self.negative_sampling(self.user_item_dict, num_nodes)
        self.user_list, self.pos = self.flatten(self.user_item_dict)
        self.len = len(self.user_list)
            
    def negative_sampling(self, user_item_dict: dict, num_nodes: tuple) -> dict:
        """
        Optimized negative sampling using numpy for set operations.
        """
        num_users, num_items = num_nodes

        all_items = np.arange(num_users, num_users + num_items)  # 아이템 ID 범위 생성
        unseen_item_dict = {}
        for user in user_item_dict:
            seen_items = np.array(user_item_dict[user])  # 사용자가 본 아이템
            unseen_items = np.setdiff1d(all_items, seen_items)  # 보지 않은 아이템 찾기
            unseen_item_dict[user] = unseen_items

        return unseen_item_dict
    
    def flatten(self, user_item_dict):
        """
        Flatten user-item interactions and negative samples for DataLoader.
        """
        user_list = []
        pos_list = []
        for user in user_item_dict:
            for pos in user_item_dict[user]:
                user_list.append(user)
                pos_list.append(pos)
            
        return user_list, pos_list
    
    def get_neg_smaples(self, user):
        return np.random.choice(self.unseen_item_dict[user], 1).item()
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # 각 데이터셋의 배치를 반환
        return  self.user_list[index], self.pos[index], self.get_neg_smaples(self.user_list[index]) 
    
    def get_seen_nodes(self):
        return self.user_item_dict

class EvalDatasetClass(data.Dataset):
    def __init__(self, direction, edge, label, num_nodes: tuple, device='cpu') -> None:
        self.device = device
        self.num_nodes = num_nodes
        self.user_item_dict = edgelist_to_user_item_dict(edge, direction, num_nodes)
        self.map_dict = self.mapping(self.user_item_dict)
        self.len = len(self.map_dict)     
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        idx = self.map_dict[index]
        items = self.user_item_dict[idx]
        user = idx
        return user, items
    
    def mapping(self, user_item_dict):
        map_dict = {}
        for i in user_item_dict:
            map_dict[len(map_dict)] = i
        return map_dict
    
