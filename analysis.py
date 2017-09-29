import argparse, os, cPickle, sys, numpy, ntpath
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
import io
import os
import shutil
import ntpath
import numpy
import cPickle
import glob
from scipy.fftpack import fft

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-f' , '--foldPath', nargs=1, required=True, help="path to the root of the folds")  
    parser.add_argument('-m' , '--modeltype', nargs=1, required=True, help="model type")    
    parser.add_argument("-p", "--classifierParam", type=float, default=1, help="classifier parameter")
    args = parser.parse_args()
    return args

def computePreRec(CM, classNames):                                                                                                                          # recall and precision computation from confusion matrix
    numOfClasses = CM.shape[0]
    if len(classNames) != numOfClasses:
        print "Error in computePreRec! Confusion matrix and classNames list must be of the same size!"
        return
    Precision = []
    Recall = []
    F1 = []    
    for i, c in enumerate(classNames):
        Precision.append(CM[i,i] / (numpy.sum(CM[:,i])+0.001)) 
        Recall.append(CM[i,i] / (numpy.sum(CM[i,:])+0.001))
        F1.append( 2 * Precision[-1] * Recall[-1] / (Precision[-1] + Recall[-1]+0.001))
    return Recall, Precision, F1

def spectralCentroid(X):
    """Computes spectral centroid of frame (given abs(FFT))"""
    L = X.shape[0]
    ind = (numpy.arange(1, len(X) + 1)) * (100/(2.0 * len(X)))
    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + 0.000000001
    # Centroid:
    C = (NUM / DEN)
    return C

def stSpectralRollOff(X, c):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + 0.00000001
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)

def fileFeatureExtraction(fileName, signal_type):                                                                                                                         # feature extraction from file
    b = numpy.load(fileName)
    rawData = b[signal_type].astype("float64") 
    means = rawData.mean(axis = 0)                                                                                                                           # compute average
    stds = rawData.std(axis = 0)                                                                                                                             # compute std
    maxs = rawData.max(axis = 0)                                                                                                                             # compute max values
    mins = rawData.min(axis = 0)                                                                                                                             # compute min values
    centroid = []
    rolloff = []
    for f in range(rawData.shape[1]):                                                                                                                        # compute spectral features
        fTemp = abs(fft(rawData[:,f]));                                                                                                                      # compute FFT
        fTemp = fTemp[0:int(fTemp.shape[0]/2)]                                                                                                               # get the first symetrical FFT part
        c = 0.9999
        centroid.append(spectralCentroid(fTemp))                                                                                                             # compute spectral centroid
        rolloff.append(stSpectralRollOff(fTemp, c))                                                                                                          # compute spectral rolloff
    featureVector = numpy.concatenate((means, stds, maxs, mins, centroid, rolloff))                                                                          # concatenate features to form the final feature vector
    return featureVector

def dirFeatureExtraction(dirNames,signal_type):                                                                                                                          # extract features from a list of directories
    features = []
    classNames = []
    c1 = 0
    for d in dirNames:                                                                                                                                       # for each direcotry
        types = ('*.npz',)
        filesList = []
        for files in types:
            filesList.extend(glob.glob(os.path.join(d, files)))
        filesList = sorted(filesList)
        for i, file in enumerate(filesList):                                                                                                                 # for each npz file
            fv = fileFeatureExtraction(file,signal_type)   
            if numpy.isnan(fv).any():
                #print file.split('_')
                #c1+=1
                continue                                                                                                              # extract features and append to feature matrix:
            if i==0:
                allFeatures = fv
            else:
                allFeatures = numpy.vstack((allFeatures, fv))
        features.append(allFeatures)
        classNames.append(d.split(os.sep)[-1])
    #print c1
    #sys.exit()
    return classNames, features

def main(rootName,modelType,classifierParam,signal_type):        
    CMall = numpy.zeros((2,2))
    if modelType != "svm" and modelType != "svm_rbf":
        C = [int(classifierParam)]    
    else:
        C = [(classifierParam)]        
    F1s = []
    Accs = []
    for ifold in range(0, 10):                                                                                                                              # for each fold
        dirName = rootName + os.sep + "fold_{0:d}".format(ifold)                                                                                            # get fold path name
        classNamesTrain, featuresTrain = dirFeatureExtraction([os.path.join(dirName, "train", "fail"), os.path.join(dirName, "train", "success")],signal_type)          # TRAINING data feature extraction  
        bestParam = aT.evaluateClassifier(featuresTrain, classNamesTrain, 2, modelType, C, 0, 0.90)                                                         # internal cross-validation (for param selection)
        classNamesTest, featuresTest = dirFeatureExtraction([os.path.join(dirName, "test", "fail"), os.path.join(dirName, "test", "success")],signal_type)              # trainGradientBoosting data feature extraction  
        [featuresTrainNew, MEAN, STD] = aT.normalizeFeatures(featuresTrain)                                                                                 # training features NORMALIZATION                        
        if modelType == "svm":                                                                                                                              # classifier training
            Classifier = aT.trainSVM(featuresTrainNew, bestParam)        
        elif modelType == "svm_rbf":
            Classifier = aT.trainSVM_RBF(featuresTrainNew, bestParam)
        elif modelType == "randomforest":
            Classifier = aT.trainRandomForest(featuresTrainNew, bestParam)
        elif modelType == "gradientboosting":
            Classifier = aT.trainGradientBoosting(featuresTrainNew, bestParam)
        elif modelType == "extratrees":
            Classifier = aT.trainExtraTrees(featuresTrainNew, bestParam)

        CM = numpy.zeros((2,2))                                                                                                                             # evaluation on testing data
        for iC,f in enumerate(featuresTest):                                                                                                                # for each class
            for i in range(f.shape[0]):                                                                                                                     # for each testing sample (feature vector)
                curF = f[i,:]                                                                                                                               # get feature vector
                curF = (curF - MEAN) / STD                                                                                                                  # normalize test feature vector
                winnerClass = classNamesTrain[int(aT.classifierWrapper(Classifier, modelType, curF)[0])]                                                    # classify and get winner class
                trueClass = classNamesTest[iC]                                                                                                              # get groundtruth class
                CM[classNamesTrain.index(trueClass)][classNamesTrain.index(winnerClass)] += 1                                                               # update confusion matrix
        CMall += CM                                                                                                                                         # update overall confusion matrix
        Recall, Precision, F1 = computePreRec(CM, classNamesTrain)                                                                                          # get recall, precision and F1 (per class)
        Acc = numpy.diagonal(CM).sum() / CM.sum()                                                                                                           # get overall accuracy
        F1s.append(numpy.mean(F1))                                                                                                                          # append average F1
        Accs.append(Acc)                                                                                                                                    # append clasification accuracy
    print        
    print "FINAL RESULTS"
    print
    print "----------------------------------"        
    print "fold\tacc\tf1"
    print "----------------------------------"        
    for i in range(len(F1s)):
        print "{0:d}\t{1:.1f}\t{2:.1f}".format(i, 100*Accs[i], 100*F1s[i])        
    Acc = numpy.diagonal(CMall).sum() / CMall.sum()                
    Recall, Precision, F1 = computePreRec(CMall, classNamesTrain)    
    print "----------------------------------"
    print "{0:s}\t{1:.1f}\t{2:.1f}".format("Avg", 100*numpy.mean(Accs), 100*numpy.mean(F1s))
    print "{0:s}\t{1:.1f}\t{2:.1f}".format("Av CM", 100*Acc, 100*numpy.mean(F1))    
    print "----------------------------------"
    print 
    print "Overal Confusion matrix:"
    aT.printConfusionMatrix(CMall, classNamesTrain)    
    print
    print "FAIL Recall = {0:.1f}".format(100*Recall[classNamesTrain.index("fail")])
    print "FAIL Precision = {0:.1f}".format(100*Precision[classNamesTrain.index("fail")])
    print "SUCCESS Recall = {0:.1f}".format(100*Recall[classNamesTrain.index("success")])
    print "SUCCESS Precision = {0:.1f}".format(100*Precision[classNamesTrain.index("success")])

    return CMall,Acc,Recall,Precision,F1


if __name__ == '__main__':
    args = parseArguments()
    rootName = args.foldPath[0]
    modelType = args.modeltype[0]
    classifierParam = args.classifierParam
    Acc,Recall,Precision,F1 = main(rootName,modelType,classifierParam)