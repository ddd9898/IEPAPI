import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import random



symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'*':21}


class seqData(Dataset):
    def __init__(self, 
                 datapath='./data/processed/trainingData_EL.csv',
                 cv_data_path='./data/processed/fivefold_val_flags(trainingData_EL).csv',
                 val_flag = False,
                 fold = 0):
        super(seqData,self).__init__()

        #Load data file
        fulldata = pd.read_csv(datapath).values.tolist()
        cv_data = pd.read_csv(cv_data_path).values.tolist()
        
        #Get train and validation
        self.data = list()
        for n,item in enumerate(cv_data):
            if val_flag and item[fold] == 1:  #Validate
                self.data.append(fulldata[n])
            elif (not val_flag) and item[fold] == 0:  #Train
                self.data.append(fulldata[n])
                
        #Load pseudo sequences
        pseudo_dir = './data/pseudoSequence(ELIM).csv' 
        temp = pd.read_csv(pseudo_dir).values.tolist()
        self.pseudoMHC_Dic = dict()
        for item in temp:
            MHCname,pseudoMHC = item[0],item[1]
            MHCname = MHCname.replace('*','').replace(':','')
            self.pseudoMHC_Dic[MHCname] = pseudoMHC
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcName = self.data[i][1].replace('*','').replace(':','')
        mhcSeq = self.pseudoMHC_Dic[mhcName]
        gt = self.data[i][2]

        #Get input
        ConcatSeq = mhcSeq + peptide.ljust(11, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        
        #Get output
        gt = float(gt)
        
        #To tensor
        ConcatSeq = torch.LongTensor(ConcatSeq)
        gt = torch.FloatTensor([gt])

        # return data
        return ConcatSeq,gt


if __name__ == '__main__':
    trainDataset = seqData(val_flag = False, fold = 0)
    valDataset = seqData(val_flag = True, fold = 0)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(trainDataset, batch_size=10, shuffle=False)
    for item in train_loader:
        a = 1
    b = 1


