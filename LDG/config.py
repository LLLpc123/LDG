import torch
class Config():
    def __init__(self,
        #global hyperparams
        
        hidden_dim = 768,
        seed = 666,
        lr = 2e-5,
        bert_lr = 2e-5,
        warm_up_ratio = 0.1   , 
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        drop_rate = 0.1    ,
        datasets = ['rest14','rest15','rest16','lap14','twitter'],#在该集合上预训练,
        activ_in_biaffine = None,

        #parsing and pretrain
        bert_model = 'bert-base-uncased',
        parse_batch_size = 16,
        parse_num_epoch = 5,
        pos = False,    #暂时不考虑pos
        train_bert = True,
        pretrain_method  = ['after_parse','together','no_pretrain','pretrain_only'][1],
        save_path = 'parser/', 

        #absa
        batch_size = 16,
        num_epoch = 10,
        use_rel = True,
        use_pos = False,
        parser_path = None,
        use_biaffine = True,
        dataset = 'rest15',#在该数据集上验证absa
        alpha = 0.1,
        softmax = 'gumbel_softmax',
        tau = 2,
        graph_iter = {
            # 'lambda':0.5,
            # 'eta':0.5,
            'smoothness_ratio':0.02,
            'degree_ratio':0.2, #0.2
            'sparsity_ratio':0.01 #0.01
        },

        path_biaffine = None,
        
    ):
    #global hyperparams
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.lr = lr
        self.bert_lr = bert_lr
        self.warm_up_ratio = warm_up_ratio   
        self.device = device
        self.drop_rate = drop_rate    
        self.datasets = datasets
        self.eps = 1e-8

        #parsing and pretrain
        self.bert_model = bert_model
        self.parse_batch_size = parse_batch_size 
        self.parse_num_epoch = parse_num_epoch
        self.pos = pos    #暂时不考虑pos
        self.train_bert = train_bert
        self.pretrain_method  = pretrain_method
        self.save_path = save_path

        #absa
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.use_rel = use_rel
        self.use_pos = use_pos
        self.parser_path = parser_path
        self.use_biaffine = use_biaffine
        self.dataset = dataset#在该数据集上验证absa
        self.alpha = alpha
        self.softmax = softmax
        self.tau = tau
        self.graph_iter = graph_iter

        self.path_biaffine = path_biaffine
        self.activ_in_biaffine = activ_in_biaffine
cfg = Config()