import os,sys
import glob
import cPickle
import numpy as np
from pyMultimedia import DataClassification
from EEG_parser import SaveData2Binary
'''

Load a specific set of EEG data and their labels

'''
def LoadTrainingData_per_file(path, datatype):
    data = np.zeros((1,4))
    labels = []
    os.chdir(path)

    for file in glob.glob("*r*"+datatype):
        with open(file,"r") as f:
            x = cPickle.load(f)
        f.close
        if data.shape[0] == 1:
            data = np.copy(x)
        else:
            data = np.concatenate((data,x)) 
        if int(file.split('_')[4]) > 0: #for binary  classification (WIN vs LOSS)
            labels.extend([1 for i in range(x.shape[0])])
        else:
            labels.extend([-1 for i in range(x.shape[0])])
    return data, np.asarray(labels)

def LoadTrainingData_per_session(path, datatype):
    session_data = {}
    data = np.zeros((1,4))
    session_labels = {}
    os.chdir(path)
    file_list = [] 

    for file in glob.glob("*r*"+datatype):
        file_list.append(file)


        with open(file,"r") as f:
            signal = cPickle.load(f)
        f.close

        #data_id = userID_SessionID_
        data_id = file.split('_')[0]+'_'+file.split('_')[1]
        level_score =int(file.split('_')[4])

        if data_id not in session_data:
            session_data[data_id] = [signal]
            session_labels[data_id] = [-1 if abs(level_score+1) < abs(level_score-1) else 1]
        else:
            session_data[data_id].append(signal)
            session_labels[data_id].append(-1 if abs(level_score+1) < abs(level_score-1) else 1)

    for key in session_data:
        print len(session_labels[key])
        print session_labels[key]
    sys.exit()

'''
        if data.shape[0] == 1:
            data = np.copy(x)
        else:
            data = np.concatenate((data,x)) 
        if int(file.split('_')[4]) > 0: #for binary  classification (WIN vs LOSS)
            labels.extend([1 for i in range(x.shape[0])])
        else:
            labels.extend([-1 for i in range(x.shape[0])])
    return data, np.asarray(labels)
'''





def EEGStats(path, datatype):
    data = {}
    labels = []    
    lines_in_file = 0
    file_count = 0
    avg_num_of_recordings = 0
    file_list = []

    user_id = []
    session_id = []
    turn_id = []
    #level_id = []
    
    os.chdir(path)

    for file in glob.glob("*r*"+datatype):

         user_id[int(file.split('_')[0])]

         with open(file,"r") as f:
            signal = cPickle.load(f)
         f.close
         print file
         user_id.append(int(file.split('_')[0])).append(session_id.append(int(file.split('_')[1])))
         #session_id.append(int(file.split('_')[1]))
         turn_id.append(int(file.split('_')[3]))
         #level_id.append(abs(int(file.split('_')[4])))

         
         if int(file.split('_')[4]) > 0: #for binary  classification (WIN vs LOSS)
            labels.append(1)
         else:
            labels.append(-1)

         data.append(signal)

    #print turn_id
    sys.exit()



    for file in file_list:
        with open(file,"r") as f:
            signal = cPickle.load(f)
        f.close
        lines_in_file+=signal.shape[0]
        file_count+=1
        data.append(signal)
        if int(file.split('_')[3]) > 0: #for binary  classification (WIN vs LOSS)
            labels.extend([1 for i in range(x.shape[0])])
        else:
            labels.extend([-1 for i in range(x.shape[0])])        
            print labels,labels[-1].shape[0],data[-1].shape[0]
        sys.exit()



    #avg_num_of_recordings = float(lines_in_file)/float(file_count)
    #print avg_num_of_recordings


        if data.shape[0] == 1:
            data = np.copy(x)
        else:
            data = np.concatenate((data,x)) 
        if int(file.split('_')[3]) > 0: #for binary  classification (WIN vs LOSS)
            labels.extend([1 for i in range(x.shape[0])])
        else:
            labels.extend([-1 for i in range(x.shape[0])])
    return data, np.asarray(labels)


def TrainEGGClassifier(data,labels):
        #model, data_scaler = DataClassification.TrainSVMClassification(data, labels, 80, 3, kernel='rbf',c_values= [1,0.05],visible=True,save=True,train_all=False)
        model, data_scaler = DataClassification.TrainSVMClassification(data, labels, 80, 0, save=True,visible=True,c_values=[0.1,0.50,0.2],train_all = False,save_path="/home/michalis/poutsaras_SVC4_all",kernel="rbf")











if __name__ == "__main__":

    #ReadSessionFile(filename)
    root = 'EEG_DATA'
    data_to_load = 'Aa'
    #data, labels = LoadTrainingData(root,data_to_load)
    LoadTrainingData_per_session(root,data_to_load)
    EEGStats(root,data_to_load)
    TrainEGGClassifier(data, labels)
    