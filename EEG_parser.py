import os, sys
import numpy as np
import cPickle
from fnmatch import fnmatch
import glob, os


filename = '/home/michalis/Documents/sequence-learning/data/user_1/session_3/robot_1'


def getUsers(root):
    user_id = []
    for f in os.listdir(root):
        user_id.append(f.split('/')[-1].split('_')[-1])
    return user_id


'''

EEG file parser

'''
def ReadSessionFile(filename):
    data = {}
    data['h_eeg'] = []
    data['h'] = [[1,1,1,1]]


    with open(filename,"r") as f:
        txt = f.readlines()
    f.close
    
    for line in txt:
        key = line.split()[0]
        value = line.split()[1:]    
        if key not in data.keys():
            data[key] = []
        if key == 'eeg':
            data['h'][-1]
            data['h_eeg'].append(data['h'][-1])
        data[key].append(value)  
    data['h'] = np.delete(data['h'],0,0)    
    return data    



def main(argv):

    folds = 10
    root = argv[0]
    train_percent = 0.8

    user_list = getUsers(root)

    for i in range(folds):
         train_success = ('/').join(('EEG_DATA','fold_'+str(i),'train','success'))
         train_fail=('/').join(('EEG_DATA','fold_'+str(i),'train','fail'))
         test_success = ('/').join(('EEG_DATA','fold_'+str(i),'test','success'))
         test_fail=('/').join(('EEG_DATA','fold_'+str(i),'test','fail'))

         if not os.path.exists(train_success):
            os.makedirs(train_success)
         if not os.path.exists(train_fail):
            os.makedirs(train_fail)
         if not os.path.exists(test_success):
            os.makedirs(test_success)
         if not os.path.exists(test_fail):
            os.makedirs(test_fail)

         user_list_randperm = np.random.permutation(user_list)
         train_users = user_list_randperm[:int(train_percent*len(user_list))]
         test_users = user_list_randperm[int(train_percent*len(user_list)):]
         
         print
         print i
         print user_list_randperm
         print  train_users,test_users

         #load eeg data for each user
         for user in user_list_randperm:      
            for session in os.listdir(('/').join((root,"user_"+user))):      

                #read logfile and load labels for all turns in the session
                with open(('/').join((root,"user_"+user,session,"logfile")),"r") as f:
                    logdata = f.readlines()
                f.close

                #for every round of this session
                for j,line in enumerate(logdata):
                    # get WIN-LOSE label
                    label = line.split(" ")[4] 
                    #open the EEG_robot file that corresponds to the ith round of the session
                    data = ReadSessionFile(('/').join((root,"user_"+user,session,"robot_"+str(j+1))))
                    if user in train_users: # if  user belongs to training in this fold
                        if label == '1':
                            save = ('/').join((train_success,user+'_'+session.split('_')[-1]+'_'+str(j+1)))
                        else:
                            save = ('/').join((train_fail,user+'_'+session.split('_')[-1]+'_'+str(j+1)))

                    else: #if user belongs to testing
                         if label == '1':
                            save = ('/').join((test_success,user+'_'+session.split('_')[-1]+'_'+str(j+1)))

                         else:
                            save = ('/').join((test_fail,user+'_'+session.split('_')[-1]+'_'+str(j+1)))

                    for key in data.keys():
                        data[key] = np.array(data[key])
                    np.savez(save,h=data['h'],h_eeg=data['h_eeg'],c=data['c'],raw=data['eeg'],
                             a=data['a'], b=data['b'], g=data['g'], d=data['d'], t=data['t'] ,
                             Aa=data['Aa'], Ab=data['Ab'], Ag=data['Ag'], Ad=data['Ad'], At=data['At'],
                             ascore=data['as'], bscore=data['bs'],gscore=data['gs'],dscore=data['ds'],tscore=data['ts'])




if __name__ == "__main__":
    #ReadSessionFile(filename)
    main(sys.argv[1:])
    #getUsers(root)
    #LoadEEGDirs(root)