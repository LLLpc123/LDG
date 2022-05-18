
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.parser import ConlluReader, collate_and_pad4parse
from utils.global_func import DataSet
from utils.absa import ABSADatesetReader
from utils.pretrain import collate_and_pad4pt
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(cfg,model,optimizer,scheduler,parse_data,pretrain_dataset,len_train_data):
    train_loader,test_loader,val_loader = parse_data
    for i in range(cfg.parse_num_epoch):
        total = 0
        arc_correct_total = 0
        rel_correct_total = 0
        pos_correct_total = 0
        total_loss = 0
        if cfg.pretrain_method != 'pretrain_only':
            for batch_idx, data in enumerate(tqdm(train_loader)):
                model.train()
                
                
                #dependency parsing
                inputs = (data['word_ids'].to(cfg.device), data['attention_mask'].to(
                    cfg.device), data['token_start_idxs'].to(cfg.device), data['seq_len'].to(cfg.device))
                arc_label = data['head'].to(cfg.device)
                rel_label = data['deprel'].to(cfg.device)
                word_len = data['seq_len'].to(cfg.device)
                pos = data['pos'].to(cfg.device)

                arc_score, rel_score,last_pooled_hidden_state = model(inputs)
                loss,arc_correct, rel_correct,pos_correct = model.loss(arc_score, rel_score,last_pooled_hidden_state,
                                arc_label, rel_label,pos, word_len-2)
                if cfg.pretrain_method == 'together':
                    # bs_pretrain = len(pretrain_dataset)//(len_train_data//cfg.parse_batch_size +1)
                    bs_pretrain = cfg.batch_size

                    if batch_idx*bs_pretrain >= len(pretrain_dataset):
                        if (batch_idx*bs_pretrain+bs_pretrain)%len(pretrain_dataset) <(batch_idx*bs_pretrain)%len(pretrain_dataset):
                            batch_pretrain_data = collate_and_pad4pt(pretrain_dataset[(batch_idx*bs_pretrain)%len(pretrain_dataset) : -1])
                        else:
                            batch_pretrain_data = collate_and_pad4pt(pretrain_dataset[(batch_idx*bs_pretrain)%len(pretrain_dataset) : (batch_idx*bs_pretrain+bs_pretrain)%len(pretrain_dataset)])
                    else:
                        batch_pretrain_data = collate_and_pad4pt(pretrain_dataset[batch_idx*bs_pretrain : batch_idx*bs_pretrain+bs_pretrain])
                    
                    token_idx = batch_pretrain_data['token_idx'].to(cfg.device)
                    attention_mask = batch_pretrain_data['attention_mask'].to(cfg.device)
                    mask_token_idx = batch_pretrain_data['mask_token_idx'].to(cfg.device)
                    label = batch_pretrain_data['label'].to(cfg.device)
                    outputs = model.mlm_model(input_ids = mask_token_idx,attention_mask = attention_mask, labels=label)
                    loss += outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total += torch.sum(word_len-1)
                arc_correct_total += arc_correct
                rel_correct_total += rel_correct
                pos_correct_total += pos_correct
                total_loss += loss.item()
        if cfg.pretrain_method == 'after_parse' or cfg.pretrain_method == 'pretrain_only':
            for index in range(len(pretrain_dataset)//cfg.batch_size ):
                batch_pretrain_data = collate_and_pad4pt(pretrain_dataset[index*cfg.batch_size : index*cfg.batch_size+cfg.batch_size])

                token_idx = batch_pretrain_data['token_idx'].to(cfg.device)
                attention_mask = batch_pretrain_data['attention_mask'].to(cfg.device)
                mask_token_idx = batch_pretrain_data['mask_token_idx'].to(cfg.device)
                label = batch_pretrain_data['label'].to(cfg.device)
                outputs = model.mlm_model(input_ids = mask_token_idx,attention_mask = attention_mask, labels=label)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if cfg.pretrain_method != 'pretrain_only':
            print('epoch {} in train set UAS:{}%, LAS:{}%, POS_acc:{}%, training loss:{}'
              .format(i, arc_correct_total/total*100,
                    rel_correct_total/total*100,
                    pos_correct_total/total*100,
                    total_loss/total))

            arc_acc, rel_acc,pos_acc = val(cfg,model,val_loader)
            print('epoch {} in val set UAS:{:.4f}%, LAS:{:.4f}%,POS_acc:{:.4f}'.format(
                i, arc_acc*100, rel_acc*100,pos_acc*100))

def val(cfg,model,dataloader):
    model.eval()
    total = 0
    arc_correct_total = 0
    rel_correct_total = 0
    pos_correct_total = 0
    for index,data in enumerate(dataloader):
        inputs = (data['word_ids'].to(cfg.device), data['attention_mask'].to(
            cfg.device), data['token_start_idxs'].to(cfg.device), data['seq_len'].to(cfg.device))
        arc_label = data['head'].to(cfg.device)
        rel_label = data['deprel'].to(cfg.device)
        word_len = data['seq_len'].to(cfg.device)
        pos = data['pos'].to(cfg.device)

        arc_score, rel_score,last_pooled_hidden_state = model(inputs)
        arc_correct, rel_correct,pos_correct = model.evaluate(
            arc_score, rel_score,last_pooled_hidden_state,
             arc_label, rel_label,pos, word_len-2)
        total += torch.sum(word_len-3)
        arc_correct_total += arc_correct
        rel_correct_total += rel_correct
        pos_correct_total += pos_correct
    return (arc_correct_total/total, rel_correct_total/total,pos_correct_total/total)

def parse_and_pretrain(cfg,model):
    train_data = DataSet(ConlluReader(
        r'datasets\parse\ptb_train_3.3.0.sd.clean').get_data())
    test_data = DataSet(ConlluReader(
        r'datasets\parse\ptb_test_3.3.0.sd.clean').get_data())
    val_data = DataSet(ConlluReader(
        r'datasets\parse\ptb_dev_3.3.0.sd.clean').get_data())
    train_loader = DataLoader(train_data, batch_size=cfg.parse_batch_size, 
                            collate_fn=collate_and_pad4parse, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=cfg.parse_batch_size,
                            collate_fn=collate_and_pad4parse, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=cfg.parse_batch_size,
                            collate_fn=collate_and_pad4parse, shuffle=False)

    pretrain_dataset = DataSet()
    for dataset in cfg.datasets:
        pretrain_dataset.extend(ABSADatesetReader(dataset=dataset).train_data.data)

    bert_params = ["bert.embeddings", "bert.encoder"]
    optimizer_grouped_parameters =[
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in bert_params)],
            "lr": cfg.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in bert_params)],
            "lr": cfg.lr
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr, eps=cfg.eps)

    total_step = len(train_data)//cfg.parse_batch_size * cfg.parse_num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warm_up_ratio * total_step, num_training_steps=total_step)
    train(cfg,model,optimizer,scheduler,(train_loader,test_loader,val_loader),pretrain_dataset,len(train_data))

