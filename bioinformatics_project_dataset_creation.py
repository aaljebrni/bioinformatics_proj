import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import os




def freq(lines, header):
    df = pd.DataFrame(lines)
    df.columns = header
    
    #slicing the disease states and plotting their histogram and printing the frequencies of the states within it
    ds = df[header[0]]#.Characteristics_DiseaseState
    
    ds.hist()
    ds = ds.to_list()
    frequency = {term: ds.count(term) for term in ds}
    # print('Here is the frequency of each disease state within this file:\n',frequency)
    
    return frequency, df

def labels_files_creation(file):
    
    data = pd.read_csv(file)
    cols =  data.columns
    
    for i in tqdm(range(len(cols) - 1)):
        selected = pd.DataFrame([data[cols[i]], data.Hybridization_Name])
        selected = selected.transpose()
        if os.path.exists(cols[i].strip('\n')+'_51.csv') == False:
            selected.to_csv( cols[i].strip('\n')+'_51.csv',    index = False)
    
    return cols
    





def clean_labels_files_create_DS(cols):
    for i in tqdm(range(2,3)):
        # removing rows that has empty disease state
        lines = list()
        with open(cols[i].strip("\n")+'.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                lines.append(row)
                for field in row:
                    #print(field)
                    if field == '  ':
                        lines.remove(row)
    
        header = lines[0]
        
        lines = lines[1:]
        
        # any disease state with the value of 'normal' or 'control brain' is converted to 'healthy' because they are
        for j in range(len(lines)):
            
            if lines[j][0] == 'normal' or lines[j][0]=='control brain':
                lines[j][0] = 'healthy'
            if lines[j][0] == 'femalemale':
                lines[j][0] = 'mixed_sex'
        
        df = pd.DataFrame(lines)
        df.columns = header
        
        # #slicing the disease states and plotting their histogram and printing the frequencies of the states within it
        frequency, _ = freq(lines, header)
        
        # removing disease states with freuency less tan 15
        freq_less_than_15 = []
        for f in frequency:
            if frequency[f] < 15 or frequency[f] > 300:
                freq_less_than_15.append(f)
        
        
        
        for fl in freq_less_than_15:
            j=0
            
            while(True):
                
                if lines[j][0] == fl:
                    
                    lines.remove(lines[j])
                    
                    continue
                j += 1
                if j == len(lines):
                    
                    break
       
        
        frequency, labels_df = freq(lines, header)
        
        
        
        
        count_pairs = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))
        
        classes, _ = list(zip(*count_pairs))
        classes_to_id = dict(zip(classes, range(len(classes))))
        
        f = open(cols[i]+'_classes_dict_51.txt',"a")
        f.write(str(classes_to_id))
        f.close()
        
        for k in range(len(lines)):
            
            lines[k][0]=classes_to_id[lines[k][0]]
        
        labels_df = pd.DataFrame(lines)
        labels_df.columns = header
        
        hyb_names = labels_df.Hybridization_Name
        hyb_names = hyb_names.to_list()
        
        finalize_ds_creation(cols[i], hyb_names, labels_df)






def finalize_ds_creation(col, hyb_names, labels_df):
    

    dataset_file = open('microarray.original.txt')
    i=0
    ds=[]
    for row in dataset_file:
        if i==0:
            hybs = row.strip('\n').split('\t')
            hybs[1:] = [hyb[hyb.find('(') + len('(') : hyb.find(')')] for hyb in hybs[1:]]
            ds.append(hybs)
            i+=1
            continue
        i+=1
    
        dr = row.strip('\n').split('\t')
        dr[1:] = [float(i) for i in dr[1:]]
        ds.append(dr)
        
    df = pd.DataFrame(ds)
    df = df.transpose()
    
    df=df.values
    
    
    
    
    dataset = []
    header = []
    for i in range(df.shape[0]):
        if i==0:
            header=[ h.strip("\"") for h in df[0,1:]]
        else:
                dcol =list(df[i,:])
                if dcol[0] in hyb_names:
                    # header.append(col[0])
                    dataset.append(dcol[1:])#[1:])
    
    dataset=pd.DataFrame(dataset)
    dataset.columns = header
    
    
    data = dataset.values
    
    labels_related_data = labels_df[col].values
    labels_related_data = labels_related_data.reshape(len(labels_related_data),1)
    
    data = np.c_[data, labels_related_data]
    data = pd.DataFrame(data)
    header.append(col)
    data.columns = header
    if os.path.exists( col+'_ds_51.csv') == False:
        data.to_csv( col+'_ds_51.csv', index = False)
    print(col+'_ds_51.csv is created or already there!')
    
    
    
#this files is the modeified version of "E-TABM-185-sdrf.csv" after I removed the columns I don't need manually, 
#by first parsing the "E-TABM-185-sdrf.csv" file using simple line of code which is reading the file and organizing it in a readable csv file that can be opened in excel and worked on manually (easy peasy)
# the E-TABM-185-sdrf.csv originally has around 16 columns for categories that can be used for the final dataset, but I used the 'Characteristics_DiseaseState' the others was created just for experimentation 'in case but I did not need them'
file = 'E-TABM-185-sdrf1.csv'
cols = labels_files_creation(file)    
clean_labels_files_create_DS(cols)
