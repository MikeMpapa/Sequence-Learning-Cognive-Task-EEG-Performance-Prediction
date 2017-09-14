import os, sys
import numpy as np
import cPickle


filename = '/home/michalis/Documents/sequence-learning/data/user_1/session_3/robot_1'



def SaveData2Binary(data,path,filename):
    save_dir = ('/').join((path,filename))
    with open(save_dir,'wb') as f:
        cPickle.dump(data,f)
    f.close

def ReadSessionFile(filename):
    data = {}

    with open(filename,"r") as f:
        txt = f.readlines()
    f.close
    
    for line in txt:
        key = line.split()[0]
        value = line.split()[1:]    
        if key not in data.keys():
            data[key] = []
        data[key].append(value)

    

    for key in data.keys():
        tmp = filename.split('/')[-3:]
        data_file_name = ('_').join((tmp[-3].split('_')[-1], tmp[-2].split('_')[-1], tmp[-1].split('_')[0][0] + tmp[-1].split('_')[-1], key))
        print data_file_name
        SaveData2Binary(np.asarray(data[key]), 'EEG_DATA', data_file_name)
    



if not os.path.exists('EEG_DATA'):
        os.makedirs('EEG_DATA')

ReadSessionFile(filename)

