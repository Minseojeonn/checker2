# template for data loader
import numpy as np
import torch.utils.data.dataset as Dataset
import torch

from dataset.utils import split_data, load_data
from dataset.DatasetClass import TrnDatasetClass, EvalDatasetClass


class DataTemplate(object):
    """Template for data loader
        for unsigned graph

    Args:
        model (str): Model
        dataset_name (str): dataset name
        seed (int): seed
        split_ratio (list): [train(float), val(float), test(float)], train+val+test == 1 
        dataset_shuffle (bool): dataset_shuffle if True
        device (str): device
        direction (str): True-direct, False-undirect
    """

    def __init__(
        self,
        dataset_name: str,
        seed: int,
        split_ratio: list,
        dataset_shuffle: bool,
        device: str,
        direction: bool,
        input_dim: int,
        augmentation: bool,
        iter_k: int,
        alpha: float,
        eps : float,
        sign : int
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = f"./dataset/{self.dataset_name}.txt"
        self.seed = seed
        self.split_ratio = split_ratio
        self.dataset_shuffle = dataset_shuffle
        self.device = device
        self.direction = direction
        self.input_dim = input_dim
        self.augmentation = augmentation
        self.eps = eps
        self.iter_k = iter_k
        self.alpha = alpha
        self.sign = sign
        assert np.isclose(sum(split_ratio), 1).item(
        ), "sum of split_ratio is not 1"
        self.processing()
        self.build_trainnormajd()
        self.nomalizing_graph()
        

    def processing(
        self,
    ):
        array_of_edges, self.num_nodes, self.num_edges = load_data(
            self.dataset_path, self.direction, self.sign)
        self.num_users = self.num_nodes[0]
        self.num_items = self.num_nodes[1]
        processed_dataset = split_data(
           array_of_edges ,self.split_ratio, self.seed, self.dataset_shuffle)
        processed_dataset["init_emb"] = self.set_init_embeddings() 
        self.processed_dataset = processed_dataset
        

    def get_dataset(self):
        train_dataset = TrnDatasetClass(self.direction, self.processed_dataset["train_edges"], self.processed_dataset["train_label"], self.num_nodes)
        val_dataset = EvalDatasetClass(self.direction, self.processed_dataset["valid_edges"], self.processed_dataset["valid_label"], self.num_nodes)
        test_dataset = EvalDatasetClass(self.direction, self.processed_dataset["test_edges"], self.processed_dataset["test_label"], self.num_nodes) 
        
        return train_dataset, val_dataset, test_dataset, self.num_nodes
    
    def set_init_embeddings(self):
        """
        set embeddings function for training model

        Args:
            embeddings (torch.Tensor): embeddings
        """
        self.embeddings_user = torch.nn.init.xavier_uniform_(
            torch.empty(self.num_nodes[0], self.input_dim))
        self.embeddings_item = torch.nn.init.xavier_uniform_(
            torch.empty(self.num_nodes[1], self.input_dim))
        return [self.embeddings_user, self.embeddings_item]
        
    def build_trainnormajd(self):
        self.train_data = self.processed_dataset["train_edges"]
        self.train_label = self.processed_dataset["train_label"]
        user_dim = torch.LongTensor(self.train_data[:,0])
        item_dim = torch.LongTensor(self.train_data[:,1])
           
        first_sub = torch.stack([user_dim, item_dim+self.num_users])
        second_sub = torch.stack([item_dim+self.num_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        if self.sign != -1:
            data = torch.cat([torch.LongTensor(self.train_label) ,torch.LongTensor(self.train_label)])
        else:
            data = torch.ones(index.size(-1)).int()
       
        assert (index.shape[-1] == len(data))
       
        self.adj_matrix = torch.sparse.FloatTensor(index, data, torch.Size([self.num_users+self.num_items, self.num_users+self.num_items]))
    
    def nomalizing_graph(self):
        """
            Calculate the degree of each node in the graph.
            Args:
                graph (torch.sparse): torch.sparse matrix 
                |  0   R |
                | R.T  0 |
            Returns:
                dict: degree of each node in the graph
        """
        
        dense = self.adj_matrix.to_dense()
        D = torch.sum(abs(dense), dim=1).float()
        D[D == 0.] = 1. #avoid division by zero
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense/D_sqrt.t() 
        index = dense.nonzero()
        data = dense[torch.logical_or(dense >= 1e-9, dense <= -1e-9)]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.num_items+self.num_users, self.num_items+self.num_users]))
        self.adj_matrix = Graph.coalesce()
    
    def get_adj_matrix(self):
        return self.adj_matrix
    
    def get_embeddings(self):
        return self.processed_dataset["init_emb"]