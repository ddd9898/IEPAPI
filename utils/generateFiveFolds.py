import numpy as np
import pandas as pd
import random

def fiveFoldCrossValidation(number_samples):
    '''
    Divide the data into 5-fold cross-validation
    '''
    index_list = list(range(number_samples))
    random.shuffle(index_list)
    val_num = int(number_samples/5)
    folds_val = [index_list[:val_num],index_list[val_num:2*val_num],index_list[2*val_num:3*val_num],index_list[3*val_num:4*val_num],index_list[4*val_num:]]

    return folds_val

#Code 1:Set random seed
seed = 1998
random.seed(seed)
np.random.seed(seed)

#Load training data
full_data = pd.read_csv("./data/DataS1.csv").values.tolist()
# full_data = pd.read_csv("./data/DataS2.csv").values.tolist()
val_flags = np.zeros((len(full_data),5)).astype(int)

#Find all samples
val_data_index_list = list()
for index,item in enumerate(full_data):
    EL = item[2]

    val_data_index_list.append(index)

# 5 cross validation in data
folds_val = fiveFoldCrossValidation(len(val_data_index_list))
for n,fold in enumerate(folds_val):
    for index_IM in fold:
        index = val_data_index_list[index_IM]
        val_flags[index,n] = 1

column = ['fold-1','fold-2','fold-3','fold-4','fold-5']
output_dir = './data/processed/fivefold_val_flags(DataS1).csv'
# output_dir = './data/processed/fivefold_val_flags(DataS2).csv'
output = pd.DataFrame(columns=column,data = val_flags)
output.to_csv(output_dir,index = None)




