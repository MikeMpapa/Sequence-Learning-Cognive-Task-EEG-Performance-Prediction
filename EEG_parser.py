import os, sys
import numpy as np
import cPickle
from fnmatch import fnmatch


filename = '/home/michalis/Documents/sequence-learning/data/user_1/session_3/robot_1'


'''

Save any kind of data to binary

'''
def SaveData2Binary(data,path,filename):
    save_dir = ('/').join((path,filename))
    with open(save_dir,'wb') as f:
        cPickle.dump(data,f)
    f.close


'''

EEG file parser

'''
def ReadSessionFile(filename, session_score):
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

        #userID_sessionID_robot/userTUrnID_sessionScore_dataType
        data_file_name = ('_').join((tmp[-3].split('_')[-1], tmp[-2].split('_')[-1], tmp[-1].split('_')[0][0] + tmp[-1].split('_')[-1], str(session_score),key))
        SaveData2Binary(np.asarray(data[key]), 'EEG_DATA', data_file_name)
    

'''

Read all EEG file directories 

'''
def LoadEEGDirs(root):

    pattern1 = 'robot_*'
    pattern2 = 'user_*'
    logcheck = False
    for path, subdirs, files in os.walk(root):
        
        session_scores = []
        if not logcheck:
            if os.path.exists(('/').join((path,"logfile"))):
                with open(('/').join((path,"logfile")),"r") as f:
                    logdata = f.readlines()
                    logcheck = True
                f.close
                print ('/').join((path,"logfile"))
                for line in logdata:
                    session_scores.append(line.split(" ")[3])
                
        for name in files:
            logcheck = False
            interaction_id = str(name.split('_')[-1])
    
            if fnmatch(name, pattern1) or fnmatch(name, pattern2):
                if int(interaction_id)-1 < len(session_scores): #to avoid errors during the data recording
                    ReadSessionFile (os.path.join(path, name), session_scores[int(interaction_id)-1])




if __name__ == "__main__":

    if not os.path.exists('EEG_DATA'):
            os.makedirs('EEG_DATA')

    #ReadSessionFile(filename)
    root = '/home/michalis/Documents/sequence-learning/data/'
    LoadEEGDirs(root)