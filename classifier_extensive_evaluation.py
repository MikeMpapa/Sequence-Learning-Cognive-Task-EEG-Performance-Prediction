from analysis import main as main_analisis
import numpy as np
import os,sys
import csv



if __name__ == '__main__':

   # signals = ['raw','a'] 
   # model_list = ['gradientboosting','extratrees']
   #root = '../tmp/'


    signals = ['raw','a','b','g','d','t','Aa','Ab','Ag','Ad','At','ascore','bscore','gscore','dscore','tscore'] 

    model_list = [sys.argv[1]] #['svm', 'svm_rbf','randomforest','gradientboosting','extratrees']
    root = '../EEG_ALL_DATA/'

    to_csv = []
    
    if not os.path.exists('Results/'):
            os.makedirs('Results/')
    converege = 0

    for model in model_list:               
            F1_max = 0
            best_Acc=0
            best_Rec=[0,0]
            best_Prec=[0,0]
            best_F1=[0,0]
            best_CM = np.zeros((2,2))
            best_p = 0
            best_signa = ''
            for signal in signals:
                    pre_estimation_check = False
                    best_ps={} #code_acceleration variable
                    param_step = 100
                    param_max = 300
                    param_min = 1
                    best_p_sig=0
                    F1_max_sig = 0
                    
                    best_CM_sig = np.zeros((2,2))
                    print
                    print
                    print'-*-*-*-*-*-*-*'
                    print model, signal
                    print'-*-*-*-*-*-*-*'

                    while True: 
                        if param_min == 1:
                            params =  np.arange(0,param_max+param_step,param_step)
                            params[0] = param_min
                        else:
                            params =  np.arange(param_min,param_max+param_step,param_step)
                        for p in params:
                            print
                            if p not in best_ps:
                                CM,Acc,Recall,Precision,F1 = main_analisis(root,model,p,signal)
                            else:
                                CM = best_ps[p][0]
                                Acc = best_ps[p][1]
                                Precision = best_ps[p][2]
                                Recall = best_ps[p][3]
                                F1 = best_ps[p][4]
                                pre_estimation_check = True
                            if pre_estimation_check is True:
                                print
                                print"=========================="
                                print"Pre - estimated param:",p,'F1:',np.mean(best_ps[p][4])
                                print"=========================="
                                print
                                pre_estimation_check=False
                            
                            if F1_max_sig < 100*np.mean(F1):
                                F1_max_sig=100*np.mean(F1)
                                best_p_sig = p
                                best_ps[p]=[CM,Acc,Recall,Precision,F1]



                            if F1_max < 100*np.mean(F1):
                                F1_max=100*np.mean(F1)
                                best_p=p
                                best_p_final = p
                                best_signal = signal
                                best_F1=F1
                                best_Prec=Precision
                                best_Rec=Recall
                                best_CM=CM
                                best_Acc=Acc


                       
                        param_step = int(param_step/2)
                        param_min = best_p_sig - param_step
                        if param_min <= 0:
                                param_min = 1
                        param_max = best_p_sig + param_step
                        if param_step < 5:
                                print 
                                print 
                                print"BEST CURRENT P FOR CURRENT SIGNAL",signal,"IS",best_p_sig,"\WITH STEP",param_step
                                print
                                print
                                break
                    print 
                    print 
                    print"BEST P FOR SIGNAL",signal,"IS",best_p_sig
                    print
                    print
            print
            print          
            print"BEST TOTAL P FOR BEST SIGNAL",best_signal,"IS",best_p
            print
            print
            to_csv.append([model,best_p,best_signal,best_Acc,best_Rec[0],best_Rec[1],best_Prec[0],best_Prec[1],best_F1[0],best_F1[1]])   
            np.savez('Results/'+model+'_'+str(best_p)+'_'+best_signal+'_'+'no_temporal',CM=CM,Acc=Acc,Rec=Recall,Pre=Precision,F1=F1)


    if not os.path.exists('Results/'):
        f = open('Results/no_temporal.csv', "w")
        f.close

    with open('Results/no_temporal.csv', "a") as f:
        writer = csv.writer(f)
        writer.writerow(['MODEL','BEST_P','SIGNAL','ACC','PRE_C1', 'PRE_C2','REC_C1','REC_C2','F1_C1','F1_C2'])
        for clc in to_csv:
            writer.writerow(clc)
    f.close

