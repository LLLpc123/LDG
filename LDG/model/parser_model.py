from turtle import forward
from transformers import BertForMaskedLM
import torch
import torch.nn as nn
from layer.biaffine import biaffine
from utils.global_func import tokenizer,seq_len2attention_mask

class parser(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()
        self.mlm_model = BertForMaskedLM.from_pretrained(cfg.bert_model)
        self.mlm_model.bert.resize_token_embeddings(len(tokenizer))
        self.drop = nn.Dropout(cfg.drop_rate)
        self.biaffine = biaffine(cfg)
        
        if cfg.pos:
            self.lin = nn.Linear(768,45) #len(relation.pos)
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        
    def forward(self, inputs):
        (word_ids, attention_mask, token_start_idxs, _) = inputs

        batch_size, max_seq_according_to_token_level = token_start_idxs.shape
        indices = token_start_idxs.repeat(1, 768).view(
            batch_size, -1, max_seq_according_to_token_level).transpose(2, 1).to(self.cfg.device)

        bert_out = self.mlm_model.bert(input_ids=word_ids, attention_mask=attention_mask)['last_hidden_state']
        if not self.cfg.train_bert:#如果固定bert权重，把梯度截掉
            bert_out = bert_out.detach()
        outputs = self.drop(bert_out)
        last_pooled_hidden_states = torch.gather(outputs, 1, indices)
        arc_score, rel_score = self.biaffine(last_pooled_hidden_states) 

        return arc_score, rel_score, last_pooled_hidden_states

    def pred(self,inputs):
        
        if self.cfg.pos:
            arc_score, rel_score, last_pooled_hidden_states  = self.forward(inputs)
            pos_feature = self.lin(last_pooled_hidden_states)
            return arc_score, rel_score, pos_feature, last_pooled_hidden_states
        else:
            return self.forward(inputs)
            
    def loss(self, arc_score, rel_score, last_pooled_hidden_state,\
        arc_label, rel_label,pos, word_len):

        # word_len: 每个batch的数据词汇长度，带有 [ROOT]
        mask = seq_len2attention_mask(word_len)
        region_mask = mask.eq(False)

        diag = torch.eye(arc_score.size(-1), dtype=torch.bool,
                         device=self.cfg.device).unsqueeze(0)
        arc_score = arc_score.masked_fill(diag, -float('inf'))
        arc_score = arc_score.masked_fill(region_mask.unsqueeze(1), -float('inf'))
        arc_target = arc_label.masked_fill(region_mask[:, 1:], -1)

        #add arc and deprel loss 
        arc_score = arc_score[:, 1:]
        rel_score = rel_score[:, 1:]
        loss = self.crit(arc_score.contiguous().view(-1,
                         arc_score.size(-1)), arc_target.view(-1))
        rel_score = torch.gather(rel_score, 2, arc_label.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, 45))
        loss += self.crit(rel_score.view(-1, 45).contiguous(), rel_label.view(-1))

        #compute UAS and LAS
        arc_pred = torch.argmax(arc_score,dim = -1)
        arc_correct = torch.eq(arc_pred, arc_label)*mask[:,1:]
        arc_correct_num = torch.sum(arc_correct)
        rel_pred = torch.argmax(rel_score, dim = -1).squeeze()
        rel_correct = torch.sum(torch.eq(rel_pred, rel_label)*arc_correct*mask[:,1:])
        
        pos_correct = 0 
        #pos
        if self.cfg.pos:
            last_pooled_hidden_state = last_pooled_hidden_state[:,1:,:]
            pos_score = self.lin(last_pooled_hidden_state)
            pos = pos.masked_fill(region_mask[:,1:],-1)
            loss += self.crit(pos_score.view(-1, 45).contiguous(),pos.view(-1))
            pos_correct  = torch.argmax(pos_score, dim = -1).squeeze()
            pos_correct = torch.sum(torch.eq(pos_correct, pos)*mask[:,1:])

            return loss/torch.sum(word_len-1),arc_correct_num,rel_correct,pos_correct

        return loss/torch.sum(word_len-1),arc_correct_num,rel_correct,0

    def evaluate(self,arc_score, rel_score,last_pooled_hidden_state, arc_label, rel_label,pos, word_len):  #pos if self.cfg.pos
        mask = seq_len2attention_mask(word_len)
        region_mask = mask.eq(False)

        diag = torch.eye(arc_score.size(-1), dtype=torch.bool,
                         device=self.cfg.device).unsqueeze(0)
        arc_score = arc_score.masked_fill(diag, -float('inf'))
        arc_score = arc_score.masked_fill(region_mask.unsqueeze(1), -float('inf'))

        #add arc and deprel loss ,drop root
        arc_score = arc_score[:, 1:]
        rel_score = rel_score[:, 1:]
        rel_score = torch.gather(rel_score, 2, arc_label.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, 45))

        #compute UAS and LAS
        arc_pred = torch.argmax(arc_score,dim = -1)
        arc_correct = torch.eq(arc_pred, arc_label)*mask[:,1:]
        arc_correct_num = torch.sum(arc_correct)
        rel_pred = torch.argmax(rel_score, dim = -1).squeeze()
        rel_correct = torch.sum(torch.eq(rel_pred, rel_label)*arc_correct*mask[:,1:])
        pos_correct = 0
        if self.cfg.pos:
            last_pooled_hidden_state = last_pooled_hidden_state[:,1:,:]
            pos_score = self.lin(last_pooled_hidden_state)
            pos_pred = torch.argmax(pos_score, dim = -1).squeeze()
            pos_correct = torch.sum(torch.eq(pos_pred,pos)*mask[:,1:])
        return arc_correct_num,rel_correct,pos_correct
    
    def save_model(self,path):
        self.mlm_model.bert.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save(self.biaffine.state_dict(),path+'/biaffine.bin')
        if self.cfg.pos:
            torch.save(self.lin.state_dict(),path+'/pos.bin')

    def load_model(self,path):
        
        self.mlm_model = BertForMaskedLM.from_pretrained(path)
        if self.cfg.use_biaffine:
            if self.cfg.path_biaffine is not None:
                self.biaffine.load_state_dict(torch.load(self.cfg.path_biaffine))
            else:
                self.biaffine.load_state_dict(torch.load(path+'/biaffine.bin'))

        if self.cfg.pos:
            self.lin.load_state_dict(torch.load(path+'/pos.bin'))