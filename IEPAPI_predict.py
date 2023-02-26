from models.models import Model_IM
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'*':21}


def get_args():
    parser = argparse.ArgumentParser(description='The application of baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', type=str, default='',
                        help='The input file',metavar='E')
    parser.add_argument('--output', dest='output', type=str, default='',
                        help='The output file',metavar='E')


    return parser.parse_args()
 

class new_dataset(Dataset):
    def __init__(self, data_path = './test.csv'):
        super(new_dataset,self).__init__()
        
        #Load pseudo sequences
        pseudo_dir = './data/pseudoSequence(ELIM).csv' 
        temp = pd.read_csv(pseudo_dir).values.tolist()
        self.pseudoMHC_Dic = dict()
        for item in temp:
            MHCname,pseudoMHC = item[0],item[1]
            MHCname = MHCname.replace('*','').replace(':','')
            self.pseudoMHC_Dic[MHCname] = pseudoMHC
            
        #Load data file
        fulldata = pd.read_csv(data_path).values.tolist()
        self.data = list()
        not_supported_HLA = list()
        for item in fulldata:
            mhcName =item[1].replace('*','').replace(':','')
            if mhcName not in self.pseudoMHC_Dic.keys():
                if mhcName not in not_supported_HLA:
                    print('{} is not supported'.format(mhcName))
                    not_supported_HLA.append(mhcName)
                continue
            self.data.append(item)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcName = self.data[i][1].replace('*','').replace(':','')
        mhcSeq = self.pseudoMHC_Dic[mhcName]

        #Get input
        ConcatSeq = mhcSeq + peptide.ljust(11, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        
        #To tensor
        ConcatSeq = torch.LongTensor(ConcatSeq)

        # return data
        return ConcatSeq

if __name__ == '__main__':
    
    #python IEPAPI_predict.py --input ./data/processed/DataS4.csv --output ./output/results/DataS4_by_IEPAPI.csv
    #python IEPAPI_predict.py --input ./data/processed/DataS5.csv --output ./output/results/DataS5_by_IEPAPI.csv
    
    #python IEPAPI_predict.py --input ./data/processed/DataS3.csv --output ./output/results/DataS3_by_IEPAPI.csv
    
    
    #Get argument parse
    args = get_args()
    input_file = './' + args.input
    output_file = './' + args.output
    
    if not os.path.exists(input_file):
        print("Not find {}".format(input_file))
        sys.exit()

    #Init 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testDataset = new_dataset(data_path = args.input)
    test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)

    model_dir = './output/models/'
    model_basename = 'Model-IM_fold*_index0_IM.model'
    
    models = []
    for n in range(5):
        model = Model_IM(num_encoder_layers = 1).to(device)
        model_name = model_basename.replace('*', str(n))
        model_path = model_dir + model_name
        # weights = torch.load(model_path)
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        models.append(model)

    #Test 
    total_preds_EL = torch.Tensor()
    total_preds_IM = torch.Tensor()
    for data in tqdm(test_loader):
        #Get input
        ConcatSeq = data.to(device)

        #Calculate output
        output_ave_EL = 0
        output_ave_IM = 0
        for model in models:
            model.eval()
            with torch.no_grad():
                y_EL,y_IM = model(ConcatSeq)
                y_EL = y_EL.cpu()
                y_IM = y_IM.cpu()
                output_ave_EL = output_ave_EL + y_EL
                output_ave_IM = output_ave_IM + y_IM
        output_ave_EL = output_ave_EL / len(models)
        output_ave_IM = output_ave_IM / len(models)
        total_preds_EL = torch.cat((total_preds_EL, output_ave_EL), 0)
        total_preds_IM = torch.cat((total_preds_IM, output_ave_IM), 0)

    P_EL = total_preds_EL.numpy().flatten()
    P_IM = total_preds_IM.numpy().flatten()

    #Save to local
    column=['peptide','HLA','EL score','IM score']
    results = list()
    for n in range(len(P_IM)):
        results.append([testDataset.data[n][0],testDataset.data[n][1],P_EL[n],P_IM[n]])
        
    output = pd.DataFrame(columns=column,data=results)
    output.to_csv(output_file,index = None)
