from utils.absa import ABSADatesetReader
from torch.utils.data import DataLoader
from utils.absa import collate_and_pad4absa
# from model.absa_model import ABSA
from transformers import AdamW,get_linear_schedule_with_warmup
from parse_pretrainer import parse_and_pretrain
from model.absa import ABSA
from absa_trainer import train_absa
from model.parser_model import parser

def train_model(cfg):
    absa_dataset = ABSADatesetReader(dataset=cfg.dataset)
    train_data = DataLoader(absa_dataset.train_data, batch_size=cfg.batch_size, 
    collate_fn=collate_and_pad4absa, shuffle=True)
    test_data = DataLoader(absa_dataset.test_data, batch_size=cfg.batch_size, 
    collate_fn=collate_and_pad4absa, shuffle=False)
    
    parse_model = parser(cfg).to(cfg.device)
    if cfg.parser_path is not None:
        print('load exist parser')
        parse_model.load_model(cfg.parser_path)
    else:
        print('train new parser')
        parse_and_pretrain(cfg,parse_model)
        parse_model.save_model(cfg.save_path)

    model = ABSA(cfg,parse_model).to(cfg.device)
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

    # optimizer = AdamW(model.parameters(), lr=cfg.lr,)
    total_steps = len(train_data) * cfg.num_epoch

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = cfg.warm_up_ratio * total_steps, num_training_steps = total_steps)
    acc,f1 = train_absa(cfg,model,(train_data,test_data),cfg.num_epoch,optimizer,scheduler)
    return acc,f1