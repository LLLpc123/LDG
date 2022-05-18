import torch
from sklearn.metrics import f1_score,accuracy_score
from tqdm import tqdm
from torchviz import make_dot
def train_absa(cfg,model,data,num_epoch,optimizer,scheduler):

    train_loader,test_loader = data
    # eval(cfg,model,test_loader)
    best_acc,best_f1 = 0,0
    for i in range(num_epoch):
        pred_all,label_all = [] ,[]
        correct,total_loss = 0,0
        
        for index,data in enumerate(tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()
            targets = data['polarity'].to(cfg.device)
            inputs = [data['token_idx'].to(cfg.device),
                    data['attention_mask'].to(cfg.device),
                    data['token_start_idxs'].to(cfg.device),
                    data['aspect_masks'].to(cfg.device),
                    data['aspect_subword_masks'].to(cfg.device),
                    targets]
            
            outputs,loss = model(inputs)

            loss.backward() 
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, dim=1)
            
            total_loss += loss
            correct += torch.sum(preds == targets)
            pred_all.extend(preds.cpu().numpy())
            label_all.extend(targets.cpu().numpy())

        print('epoch {}:train acc:{:.4f}%|f1 score:{:.4f}|loss:{:.4f}'\
            .format(i,accuracy_score(label_all,pred_all)*100,f1_score(label_all,pred_all, average='macro'),total_loss/len(train_loader)) )
        acc,f1 = eval(cfg,model,test_loader)

        if best_acc<acc :
            best_acc,best_f1 = acc,f1
            print('epoch:{} update best : test acc:{:.4f}%|f1 score:{:.4f}'.format(i,best_acc*100,best_f1))
        if acc>0.99:
            flag += 1
            if flag>= 5:
                print('best result: test acc:{:.4f}%  |  f1 score:{:.4f}'.format(best_acc*100,best_f1))
                return best_acc,best_f1
    print('best result: test acc:{:.4f}%  |  f1 score:{:.4f}'.format(best_acc*100,best_f1))
    return best_acc,best_f1
def eval(cfg,model,test_data):
    model.eval()
    pred_all,label_all = [] ,[]
    correct = 0
    for index,data in enumerate(test_data): 
        targets = data['polarity'].to(cfg.device)
        inputs = [data['token_idx'].to(cfg.device),
                data['attention_mask'].to(cfg.device),
                data['token_start_idxs'].to(cfg.device),
                data['aspect_masks'].to(cfg.device),
                data['aspect_subword_masks'].to(cfg.device),
                targets]
        

        outputs,_ = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        correct += torch.sum(preds == targets)
        pred_all.extend(preds.cpu().numpy())
        label_all.extend(targets.cpu().numpy())
    
    return accuracy_score(label_all,pred_all),f1_score(label_all,pred_all, average='macro')
