import pandas as pd

#Load pseudo sequences 
temp = pd.read_csv('./data/NetMHCpan4.1/MHC_pseudo.dat' ,header=None).values.tolist()
pseudoMHC_Dic = dict()
for item in temp:
    item = item[0].split(' ')
    MHCname = item[0]
    pseudoMHC = item[-1]
    if len(pseudoMHC) != 34:
        print("{} is wrong!".format(MHCname))
        continue
    
    MHCname = MHCname.replace('*','').replace(':','')
    pseudoMHC_Dic[MHCname] = pseudoMHC
    

#Load all data 
data1 = pd.read_csv("./data/processed/Data S1.csv").values.tolist()
data2 = pd.read_csv("./data/processed/Data S2.csv").values.tolist()
data3 = pd.read_csv("./data/processed/Data S3.csv").values.tolist()
data4 = pd.read_csv("./data/processed/Data S4.csv").values.tolist()
data5 = pd.read_csv("./data/processed/Data S5.csv").values.tolist()
data = data1+data2+data3+data4+data5

unique_HLA = list()
for item in data:
    HLA = item[1].replace('*','').replace(':','')
    if HLA not in unique_HLA:
        unique_HLA.append(HLA)


pseudoMHC_Dic['HLA-E01033']='YHSMYRESADTIFVNTLYLWHEFYSSAEQAYTWY'
pseudo_HLA_list = list()
for HLA in unique_HLA:
    pseudo_HLA_list.append([HLA,pseudoMHC_Dic[HLA]])

column = ['HLA','pseudoSeq']
output_dir = './data/pseudoSequence(ELIM).csv'
output = pd.DataFrame(columns=column,data = pseudo_HLA_list)
output.to_csv(output_dir,index = None)
