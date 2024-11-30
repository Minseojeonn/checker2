import numpy as np
import torch
from torch import nn

class LightGCN(torch.nn.Module):
    def __init__(self, 
                 config, 
                 dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.embedding_user = nn.Embedding.from_pretrained(dataset.get_embeddings()[0])
        self.embedding_user.weight.requires_grad = True
        self.embedding_item = nn.Embedding.from_pretrained(dataset.get_embeddings()[1])
        self.embedding_item.weight.requires_grad = True
        self.num_users, self.num_items  = self.dataset.num_nodes
        self.n_layers = self.config.num_layers           
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.get_adj_matrix().to(self.config.device)
                
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_droped = self.Graph    
        all_emb = torch.sparse.mm(g_droped, all_emb)
        embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def getEmbeddingcustom(self, users, stpos_items, wkpos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        stpos_emb = all_items[stpos_items]
        wkpos_emb = all_items[wkpos_items]
        neg_emb = all_items[neg_items]
        
        users_emb_ego = self.embedding_user(users)
        stpos_emb_ego = self.embedding_item(stpos_items)
        wkpos_emb_ego = self.embedding_item(wkpos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        return users_emb, stpos_emb, wkpos_emb, neg_emb, users_emb_ego, stpos_emb_ego, wkpos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg, label): #pos = positive items, neg = negative items // item must be one to one
        pos = pos - self.num_users
        neg = neg - self.num_users
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def sbpr_loss(self, users, pos, neg, label): #pos = positive items, neg = negative items // item must be one to one
        pos = pos - self.num_users
        neg = neg - self.num_users
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        label[label == -1] = 0.5 
        pos_scores = pos_scores * label

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def mybpr_loss(self, users, strongpos, weakpos, neg, label):
        strongpos = strongpos - self.num_users
        weakpos = weakpos - self.num_users
        neg = neg - self.num_users
        (users_emb, strongpos_emb, weakpos_emb, neg_emb,
        userEmb0, strongposEmb0, weakposEmb0, negEmb0) = self.getEmbedding(users.long(), strongpos.long(), weakpos.long(), neg.long())
        
        emb_diff = weakpos_emb - strongpos_emb
        adj_neg_emb = neg_emb + emb_diff
        pos_scores = torch.mul(users_emb, strongpos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, adj_neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = self.reg_weight * (
            torch.norm(userEmb0) ** 2 +
            torch.norm(strongposEmb0) ** 2 +
            torch.norm(negEmb0) ** 2
        )
        
        return loss, reg_loss
    
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma