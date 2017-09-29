from analysis import main as main_analisis
import numpy as np
import os
import csv



if __name__ == '__main__':

   # signals = ['raw','a'] 
   # model_list = ['gradientboosting','extratrees']
   #root = '../tmp/'


    signals = ["raw",'a','b','g','d','t','Aa','Ab','Ag','Ad','At'] 

    model_list = ['svm', 'svm_rbf','randomforest','gradientboosting','extratrees']
    root = '../EEG_ALL_DATA/'

    to_csv = []
    
    if not os.path.exists('Results/'):
            os.makedirs('Results/')

    for model in model_list:

        F1_max = 0
        best_p=0
        best_signal = ''
        for signal in signals:
            print
            print
            print'-*-*-*-*-*-*-*'
            print model, signal
            print'-*-*-*-*-*-*-*'

          
            for p in range(5 , 305, 5):
                print
                CM,Acc,Recall,Precision,F1 = main_analisis(root,model,p,signal)
                if F1_max < 100*np.mean(F1):
                    F1_max=100*np.mean(F1)
                    best_p=p
                    best_signal = signal
        to_csv.append([model,best_p,best_signal,Acc,Recall[0],Recall[1],Precision[0],Precision[1],F1[0],F1[1]])   
        np.savez('Results/'+model+'_'+str(best_p)+'_'+best_signal+'_'+'no_temporal',CM=CM,Acc=Acc,Rec=Recall,Pre=Precision,F1=F1)




with open('Results/no_temporal.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(['MODEL','BEST_P','SIGNAL','ACC','PRE_C1', 'PRE_C2','REC_C1','REC_C2','F1_C1','F1_C2'])
    for clc in to_csv:
        writer.writerow(clc)
f.close