import torch
import torch.nn as nn 
from utils.global_func import seq_len2attention_mask
from layer.gcn import GCN
import numpy as np
VERY_SMALL_NUMBER = 1e-12

class ABSA(nn.Module):
    def __init__(self,cfg,parser) :
        super().__init__()
        self.cfg = cfg
        self.parser =  parser
        self.drop = nn.Dropout(cfg.drop_rate)
        self.relation_linear = nn.Linear(45,1)
        
        self.GCN1 = GCN(cfg)
        self.GCN2 = GCN(cfg)
        self.GCN3 = GCN(cfg)
        self.GCN4 = GCN(cfg)
        self.ln = nn.LayerNorm(normalized_shape=768)
        if cfg.use_biaffine:
            self.lin = nn.Linear(cfg.hidden_dim,3)
            self.activ = nn.ReLU()
        else:
            self.lin = nn.Linear(cfg.hidden_dim,3)
        self.crit = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, input):
        input_ids,attention_mask,token_start_idxs,aspect_mask,aspect_subword_masks,targets = input
        parser_input =  input_ids, attention_mask, token_start_idxs, None
        if self.cfg.use_pos:
            arc_score,rel_score,pos_feature,word_encoding = self.parser(parser_input)
        else:
            arc_score,rel_score,word_encoding = self.parser(parser_input)
        if self.cfg.use_biaffine:

            word_level_seq_len  = torch.sum(torch.gt(token_start_idxs,0),dim = -1)
            word_level_attention_mask = seq_len2attention_mask(word_level_seq_len)
            rel_struc = self.activ(self.relation_linear(rel_score).squeeze(-1))

            rel_struc_ = row_norm_and_mask_pad2d(rel_struc,word_level_attention_mask)
            arc_struc_ = row_norm_and_mask_pad2d(arc_score,word_level_attention_mask)
            rel_struc_T = row_norm_and_mask_pad2d(rel_struc.transpose(-1,-2),word_level_attention_mask)
            arc_struc_T = row_norm_and_mask_pad2d(arc_score.transpose(-1,-2),word_level_attention_mask)
            struc = row_norm_and_mask_pad2d(rel_struc_ + arc_struc_,word_level_attention_mask)
            strucT = row_norm_and_mask_pad2d(rel_struc_T+arc_struc_T,word_level_attention_mask)

            hidden_state = self.ln(self.GCN1(word_encoding,struc)+word_encoding)
            hidden_state = self.ln(self.GCN2(hidden_state,strucT)+word_encoding)
            hidden_state = self.ln(self.GCN3(hidden_state,struc)+word_encoding)
            hidden_state = self.ln(self.GCN4(hidden_state,strucT)+word_encoding)
            
            output = self.drop(hidden_state*aspect_mask.unsqueeze(-1))
        else:
            output = self.drop(word_encoding*aspect_mask.unsqueeze(-1))
        output = output.sum(1,keepdim = True).squeeze(1)/torch.sum(aspect_mask,dim = -1).unsqueeze(-1)
        output = self.lin(output)
        loss = self.CEloss(output,targets)
        # loss = loss+self.add_batch_graph_loss(struc,word_level_seq_len,hidden_state)+self.add_batch_graph_loss(strucT,word_level_seq_len)
        return output,loss

    def CEloss(self,outputs,targets):
        return self.crit(outputs,targets)
    def add_batch_graph_loss(self, out_adj,seq_len,features=None):
        # Graph regularization from Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings
        graph_loss = 0
        for i in range(out_adj.shape[0]):
            adj = out_adj[i,:seq_len[i],:seq_len[i]]
            
            # L = torch.diagflat(torch.sum(adj, -1)) - adj
            # if features is not None:
            #     feature = features[i,:seq_len[i]]
            #     graph_loss += self.cfg.graph_iter['smoothness_ratio'] * torch.trace(torch.mm(feature.transpose(-1, -2), torch.mm(L, feature))) / int(np.prod(adj.shape))
            
            #下面两个公式与原公式相反，用来在固定度矩阵的情况下提高邻接矩阵的稀疏性
            # ones_vec = torch.ones(adj.shape[-1],device = self.cfg.device)
            # graph_loss += self.cfg.graph_iter['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(0), torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).sum() / adj.shape[0]
            graph_loss = -self.cfg.graph_iter['sparsity_ratio'] * torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))
        
        graph_loss = graph_loss/out_adj.shape[0]
        # ones_vec = torch.ones(out_adj.shape[:-1],device = self.cfg.device)
        # graph_loss += -self.cfg.graph_iter['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / out_adj.shape[-1]
        # graph_loss += self.cfg.graph_iter['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss


def mask_pad2d(adj,attention_mask):
    #attention_mask : [1,1,1,1,1,0,0,0,0]
    A = adj * attention_mask[:,:,None] * attention_mask[:,None,:]
    self_loop_for_pad = torch.eye(adj.shape[-1],device = adj.device).unsqueeze(0) * VERY_SMALL_NUMBER
    return A + self_loop_for_pad 

def row_norm(adj):
    return adj/torch.sum(adj,dim = -1).unsqueeze(-1)

def row_norm_and_mask_pad2d(adj,attention_mask):
    A = adj * attention_mask[:,:,None] * attention_mask[:,None,:]
    self_loop_for_pad = torch.eye(adj.shape[-1],device = adj.device).unsqueeze(0) * VERY_SMALL_NUMBER
    A = A + self_loop_for_pad 
    A = A/torch.sum(A,dim = -1).unsqueeze(-1)
    A = A * attention_mask[:,:,None] * attention_mask[:,None,:]
    A = A + self_loop_for_pad 
    return A
