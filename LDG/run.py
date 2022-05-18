# from crypt import methods
from config import Config
from main import train_model
import torch
cfg = Config(
            
            parse_num_epoch=3,
            datasets = ['rest14','rest15','rest16','lap14'],
            activ_in_biaffine= torch.nn.ReLU(),
            save_path='params/LDG/',
        )
train_model(cfg)

with open('result.txt','a',encoding='utf-8') as f:
    f.write('params/relu_in_biaffine/  '+'\n')
for dataset in ['rest14','rest15','rest16','lap14']:
    for seed in range(666,668):
        cfg = Config(
            parser_path = 'params/LDG/',
            seed = seed,
            num_epoch=10,
            dataset = dataset,
            datasets = ['rest14','rest15','rest16','lap14'],
            activ_in_biaffine= torch.nn.ReLU()
        ) 
        acc,f1 = train_model(cfg)
        with open('result.txt','a',encoding='utf-8') as f:
            f.write(str((dataset,seed,acc,f1))+'\n')
