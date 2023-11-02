import csv
import os
import pandas as pd
PATH='/home/xinyue/kinetics-dataset/k400/'
SPLITS = ['test', 'train', 'val']

for split in SPLITS:
    csv_path=PATH+'datasets'+'/'+split +'.csv'
    file_path=PATH+split+'/'
    file_list=os.listdir(file_path)
    i=0
    label = []
    for_pd=[]
    for file in file_list:
        i=i+1
        s=file.split('.')
        if file.split('.')[-1]!='mp4':
            label.append(file)
    for l in label:
        for per_file in os.listdir(file_path+l):
            per_dict = {}
            per_dict['label']=l
            per_dict['path']=file_path+l+'/'+per_file
            for_pd.append(per_dict)
    df=pd.DataFrame(for_pd)
    df.to_csv(csv_path,header=True)
    print(1)
