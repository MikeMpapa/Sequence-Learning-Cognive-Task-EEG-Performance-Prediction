from analysis import main as main_analisis
import numpy as np


if __name__ == '__main__':

    model_list = ['svm', 'svm_rbf','randomforest','gradientboosting','extratrees']
    root = 'v'


    for model in model_list:
        print model
        F1_max = 0
        best_p=0
        for p in range(2 , 302, 2):
            Acc,Recall,Precision,F1 = main_analisis(root,model,p)
            if F1_max < 100*np.mean(F1):
                F1_max=100*np.mean(F1)
                best_p=p
     
        np.savez('Results/'+model+'_'+str(best_p),Acc=Acc,Rec=Recall,Pre=Precision,F1=F1)
     