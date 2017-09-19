import os
import glob
import cPickle
import numpy as np
from pyMultimedia import DataClassification

'''

Load a specific set of EEG data and their labels

'''
def LoadTrainingData(path, datatype):
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
        if int(file.split('_')[3]) > 0: #for binary  classification (WIN vs LOSS)
            labels.extend([1 for i in range(x.shape[0])])
        else:
            labels.extend([-1 for i in range(x.shape[0])])
    return data, np.asarray(labels)





def TrainEGGClassifier(data,labels):
        #model, data_scaler = DataClassification.TrainSVMClassification(data, labels, 80, 3, kernel='rbf',c_values= [1,0.05],visible=True,save=True,train_all=False)
        model, data_scaler = DataClassification.TrainSVMClassification(data, labels, 80, 1, save=True,visible=True,c_values=[0.1,0.50,0.2],train_all = False,save_path="/home/michalis/poutsaras_SVC4_all",kernel="rbf")











if __name__ == "__main__":

    #ReadSessionFile(filename)
    root = 'EEG_DATA'
    data_to_load = 'Aa'
    data, labels = LoadTrainingData(root,data_to_load)
    TrainEGGClassifier(data, labels)
    