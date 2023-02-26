import numpy as np
import torch
import os
import random
from models.models import baseline
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
import logging
from utils.seqdataloader import seqData
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

#CUDA_VISIBLE_DEVICES=0 python  main_train_baseline_IM.py   --fold 0   --index 0  &

def train(model, device, train_loader, optimizer, epoch):
    '''
    training function at each epoch
    '''
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    logging.info('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #Get input
        ConcatSeq = data[0].to(device)

        #Calculate output
        optimizer.zero_grad()
        y = model(ConcatSeq)
        y = y.squeeze()

        ###Calculate loss
        g = data[1].to(device).squeeze()
        # loss = SingleTaskLoss_pretrain(y,g,pretrain=False)
        loss = F.binary_cross_entropy(y,g)

    
        train_loss = train_loss + loss.item()

        #Optimize the model
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_loss

def predicting(model, device, loader):
    model.eval()
    preds = torch.Tensor()
    labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            #Get input
            ConcatSeq = data[0].to(device)

            #Calculate output
            g = data[1]
            y = model(ConcatSeq)

         
            preds = torch.cat((preds, y.cpu()), 0)
            labels = torch.cat((labels, g), 0)

    return labels.numpy().flatten(),preds.numpy().flatten()


def evalute(G,P):

    ######evalute EL
    GT_labels = list()
    Pre = list()
    for n,item in enumerate(G):
        if (not np.isnan(item)):
            GT_labels.append(int(item))
            Pre.append(P[n])

    AUC_ROC = roc_auc_score(GT_labels,Pre)
    precision_list, recall_list, _ = precision_recall_curve(GT_labels, Pre)
    AUC_PR = auc(recall_list, precision_list)

    Thresh = 0.5
    pre_labels = [1 if item>Thresh else 0 for item in Pre]

    accuracy = accuracy_score(GT_labels,pre_labels)
    recall = recall_score(GT_labels,pre_labels)
    precision = precision_score(GT_labels,pre_labels)
    F1_score = f1_score(GT_labels,pre_labels)
    
    evaluation = [accuracy,precision,recall,F1_score,AUC_ROC,AUC_PR] 
    

    return evaluation

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fold', dest='fold', type=int, default=0,
                        help='Number of fold',metavar='E')
    parser.add_argument('--index', dest='index', type=int, default=0,
                        help='Number of fold',metavar='E')


    return parser.parse_args()


def LoadDataset(fold = 0):
    '''
    Load training dataset and  validation dataset.
    Output:
        trainDataset, valDataset
    '''
    #Load Train and Val Data
    trainDataset = seqData(datapath='./data/processed/DataS2.csv',
        cv_data_path='./data/processed/fivefold_val_flags(DataS2).csv',val_flag = False, fold = fold)
    valDataset = seqData(datapath='./data/processed/DataS2.csv',
        cv_data_path='./data/processed/fivefold_val_flags(DataS2).csv',val_flag = True, fold = fold)

    return trainDataset, valDataset


def seed_torch(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


if __name__ == '__main__':
    #Fix random seed and reproduce expermental results
    seed_torch()

    #Train setting
    BATCH_SIZE = 320 #2560
    LR = 0.00005 # 0.001
    LR_PRETRAIN = 0.001 # 0.001
    LOG_INTERVAL = 3000 #300
    LOG_INTERVAL_PRETRAIN = 50
    NUM_EPOCHS = 200 #500

    #Get argument parse
    args = get_args()

    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    model_name = 'baseline-IM'
        
    add_name = '_fold' + str(args.fold) + '_index' + str(args.index)
    model_file_name =  './output/models/' + model_name + add_name
    result_file_name = './output/results/result_' + model_name + add_name + '.csv'
    
    
    logfile = './output/log/log_' + model_name + add_name + '.txt'
    fh = logging.FileHandler(logfile,mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    #Tensorboard
    logfile = './output/log/log_' + model_name + add_name
    writer = SummaryWriter(logfile)
    

    #Step 1:Prepare dataloader
    trainDataset, valDataset = LoadDataset(fold = args.fold)
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)


    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = baseline(num_encoder_layers = 1).to(device) #1

    #Step 3: Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.01

                                                    
    logging.info(f'''Starting training:
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LR}
    Training size:   {len(trainDataset)}
    Validation size: {len(valDataset)}
    Device:          {device.type}
    ''')

    best_AUCROC = -1

    early_stop_count = 0
    for epoch in range(NUM_EPOCHS):
        #Train
        train_loss = train(model, device, train_loader, optimizer, epoch)
        # scheduler.step()
        
        #Validate
        logging.info('predicting for valid data')
        G,P = predicting(model, device, valid_loader)
        # loss,precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR = evalute(G,P)
        [accuracy,precision,recall,F1_score,AUC_ROC,AUC_PR]  = evalute(G,P)

        #Logging
        logging.info('Epoch {} accuracy={:.4f},precision={:.4f},recall={:.4f},F1_score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}'.format(epoch,
    accuracy,precision,recall,F1_score,AUC_ROC,AUC_PR))
        
        
        if(best_AUCROC < AUC_ROC):
            best_AUCROC = AUC_ROC
            BestEpoch = epoch
            early_stop_count = 0

            #Save model
            torch.save(model.state_dict(), model_file_name + '_IM' +'.model')
    
        else:
            early_stop_count = early_stop_count + 1
            
            

        logging.info('BestEpoch={}; Best AUCROC={:.4f}.'.format(
            BestEpoch,best_AUCROC
        ))

        #Tensorboard
        writer.add_scalar('accuracy_val', accuracy, epoch)
        writer.add_scalar('AUC_ROC_val', AUC_ROC, epoch)
        writer.add_scalar('recall_val', recall, epoch)
        writer.add_scalar('precision_val', precision, epoch)
        writer.add_scalar('F1_score_val', F1_score, epoch)
        writer.add_scalar('AUC_PR_val', AUC_PR, epoch)
        writer.add_scalar('loss_train', train_loss, epoch)

        if early_stop_count >= 50:
            logging.info('Early Stop.')
            break


            


            

        

        







