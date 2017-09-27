from analysis import main as main_analisis
import numpy as np



if __name__ == '__main__':


    signal = ['a','b','g','d','t','Aa','Ab','Ag','Ad','At','ascore','bscore','gscore','dscore','tscore'] 
    model_list = ['svm', 'svm_rbf','randomforest','gradientboosting','extratrees']
    root = 'v'


    for model in model_list:
        best_signal = ''
        for signal in signal:
            print'-----------'
            print model, signal
            print'-----------'

            F1_max = 0
            best_p=0
            for p in range(5 , 15, 5):
                Acc,Recall,Precision,F1 = main_analisis(root,model,p,signal)
                if F1_max < 100*np.mean(F1):
                    F1_max=100*np.mean(F1)
                    best_p=p
                    best_signal = signal
     
        np.savez('Results/'+model+'_'+str(best_p)+'_'+best_signal,Acc=Acc,Rec=Recall,Pre=Precision,F1=F1)
     