from models.models import Model_atten_score
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from weblogo import *
import matplotlib
matplotlib.use('Agg')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'*':21}

AA_pos_dict = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
                            'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20}

def get_args():
    parser = argparse.ArgumentParser(description='Immune epitope Motif from model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MHC', dest='MHC', type=str, default='',
                        help='The MHC name',metavar='E')
    parser.add_argument('--MHCseq', dest='MHCseq', type=str, default='',
                        help='The MHC sequence',metavar='E')
    parser.add_argument('--require_pdf', dest='require_pdf', type=str, default='False',
                        help='save pdf or not',metavar='E')


    return parser.parse_args()

class randomPepData(Dataset):
    def __init__(self, data_path = './data/uniprot/9_peptides.csv',mhcSeq = ''):
        super(randomPepData,self).__init__()

        #Load data file
        self.data = pd.read_csv(data_path).values.tolist()
        self.mhcSeq = mhcSeq
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcSeq = self.mhcSeq

        #Get input
        ConcatSeq = mhcSeq + peptide.ljust(11, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        
        #To tensor
        ConcatSeq = torch.LongTensor(ConcatSeq)

        # return data
        return ConcatSeq


if __name__ == '__main__':
    #python IEPAPI_motif.py --MHC HLA-A*11:01 --MHCseq YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY --require_pdf True
    #python IEPAPI_motif.py --MHC HLA-B*40:01 --MHCseq YHTKYREISTNTYESNLYLRYNYYSLAVLAYEWY --require_pdf True
    #python IEPAPI_motif.py --MHC HLA-B*57:03 --MHCseq YYAMYGENMASTYENIAYIVYNYYTWAVLAYLWY --require_pdf True
    #python IEPAPI_motif.py --MHC HLA-A*68:01 --MHCseq YYAMYRNNVAQTDVDTLYIMYRDYTWAVWAYTWY --require_pdf True
    #python IEPAPI_motif.py --MHC HLA-B*44:02 --MHCseq YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY --require_pdf True
    
    #CUDA_VISIBLE_DEVICES=0 python IEPAPI_motif.py --MHC HLA-B*27:05 --MHCseq YHTEYREICAKTDEDTLYLNYHDYTWAVLAYEWY --require_pdf True &
    #CUDA_VISIBLE_DEVICES=0 python IEPAPI_motif.py --MHC HLA-B*27:09 --MHCseq YHTEYREICAKTDEDTLYLNYHHYTWAVLAYEWY --require_pdf True &
    
    #CUDA_VISIBLE_DEVICES=7 python IEPAPI_motif.py --MHC HLA-B*27:04 --MHCseq YHTEYREICAKTDESTLYLNYHDYTWAELAYEWY --require_pdf True &
    #CUDA_VISIBLE_DEVICES=7 python IEPAPI_motif.py --MHC HLA-B*27:06 --MHCseq YHTEYREICAKTDESTLYLNYDYYTWAELAYEWY --require_pdf True &
    
    #Get argument parse
    args = get_args()
    MHC =  args.MHC
    MHCSeq = args.MHCseq
    
    # MHC =  'HLA-A*11:01'
    # MHCSeq = 'YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY'
    
    #Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    #Load Data
    testDataset = randomPepData(mhcSeq = MHCSeq)
    test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)


    model_dir = './output/models/'
    model_basename = 'Model-IM_fold*_index0_IM.model'
    models = []
    for n in range(5):
        model = Model_atten_score(num_encoder_layers = 1).to(device)
        model_name = model_basename.replace('*', str(n))
        model_path = model_dir + model_name
        # weights = torch.load(model_path)
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        models.append(model)
    
    attn_output_weights = torch.Tensor()
    total_preds_EL = torch.Tensor()
    total_preds_IM = torch.Tensor()
    #Test 
    for data in tqdm(test_loader):
        #Get input
        ConcatSeq = data.to(device)
        
        output_ave_attention = 0
        output_ave_EL = 0
        output_ave_IM = 0
        for model in models:
            model.eval()
            with torch.no_grad():
                ave_attention_weight,y_EL,y_IM = model(ConcatSeq)
                
                ave_attention_weight = ave_attention_weight.cpu()
                y_EL = y_EL.cpu()
                y_IM = y_IM.cpu()
                
                output_ave_EL = output_ave_EL + y_EL
                output_ave_IM = output_ave_IM + y_IM
                output_ave_attention = output_ave_attention + ave_attention_weight
           
        output_ave_attention = output_ave_attention / len(models)
        output_ave_EL = output_ave_EL / len(models)     
        output_ave_IM = output_ave_IM / len(models)
        
        attn_output_weights = torch.cat((attn_output_weights, output_ave_attention), 0)
        total_preds_EL = torch.cat((total_preds_EL, output_ave_EL), 0)
        total_preds_IM = torch.cat((total_preds_IM, output_ave_IM), 0)
        

    attn_output_weights = attn_output_weights.numpy()  #45*45
    P_EL = total_preds_EL.numpy().flatten()
    P_IM = total_preds_IM.numpy().flatten()


    atten_data = list()
    for motif_name in ['EL','IM']:
        title_heatmap =  MHC + '_heatmap(' + motif_name + ')'
        title_logo =  MHC + '_logo(' + motif_name + ')'

        #Find top 1000
        TOP_NUM = 1000
        #Find top
        if motif_name == 'EL':
            top_indexs = np.argsort(1-P_EL)[:TOP_NUM]
        elif motif_name == 'IM':
            top_indexs = np.argsort(1-P_IM)[:TOP_NUM]
            
        top_scores = [total_preds_IM[idx] for idx in top_indexs]
        top_attens = [attn_output_weights[idx] for idx in top_indexs]

        #####Draw heatmap
        atten_scores = list()
        peptide_list = list()
        for n in range(TOP_NUM):
            idx = top_indexs[n]
            peptide = testDataset.data[idx][0]
            HLA = MHC
            
            peptide_list.append(peptide)
            
            attn_weight = np.sum(top_attens[n],axis=0)[34:]
            attn_weight = attn_weight/np.sum(attn_weight)
            
            writeLine = [peptide,HLA]
            writeLine.extend(attn_weight)
            atten_scores.append(writeLine)

        heatMapData = np.zeros((20,9))
        for item in atten_scores:
            peptide = item[0]
            if 'X' in peptide:
                continue
            atten_scores = item[2:11]
            for col_index in range(len(peptide)):
                symbol = peptide[col_index]
                row_index = AA_pos_dict[symbol] - 1
                
                heatMapData[row_index,col_index] += atten_scores[col_index]
        atten_data.append(heatMapData)

        #Draw
        plt.figure(figsize=(5, 8), dpi=150)
        ax = sns.heatmap(heatMapData,cmap='hot', yticklabels=AA_pos_dict.keys()) #Greens_r,coolwarm
        ax.set_title(title_heatmap)
        ax.set_ylabel('Amino-acid type of peptides')
        ax.set_xticklabels([str(item) for item in range(1,10)])
        plt.xticks(fontsize=8)
        plt.yticks(rotation=0,fontsize=8)
        # plt.pause(1)
        plt.savefig('./' + title_heatmap.replace('*','').replace(':','') + '.jpg',bbox_inches = 'tight')  # 保存图片
        if args.require_pdf == 'True':
            plt.savefig('./' + title_heatmap + '.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片
        plt.clf()
        plt.close()
        
        
        ##Draw logo
        f = open('./data/temp.txt','w')
        f.write('\n'.join(peptide_list))
        f.close()
        
        f = open('./data/temp.txt')
        seqs = read_seq_data(f)
        data = LogoData.from_seqs(seqs)

        options = LogoOptions()
        options.fineprint = MHC + '(' + motif_name + ')'
        # options.xaxis_label = 'Position'
        options.show_xaxis = True
        options.resolution = 330
        options.number_interval = 1
        format = LogoFormat(data,options)

        output = jpeg_formatter(data,format)
        output2 = pdf_formatter(data,format)

        with open(title_logo + '.jpg','wb') as f:
            f.write(output)
        f.close()
        # if args.require_pdf == 'True':
        #     with open(title_logo + '.pdf','wb') as f:
        #         f.write(output2)
        #     f.close()
        os.remove('./data/temp.txt')


        








