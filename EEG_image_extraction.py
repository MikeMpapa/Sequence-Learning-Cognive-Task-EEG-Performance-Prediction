import sys,os
import numpy as np
from scipy import misc
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler



def Convert2Image(path,data,h,w):
    #solution 1
     
     eeg=np.load(path+'/'+data)
     for f in eeg:
        if f.find('score') != -1 or f.find('h') != -1:
            continue

        feature = [[float(j) for j in i] for i in eeg[f]]
        scaler = MinMaxScaler()
        scaler.fit(feature)
        scaled_feature = scaler.transform(feature) #scaled according to user's min
        im = misc.imresize(scaled_feature,(h,w),interp="bicubic",mode='P')
        print ('/').join((path,f)),"---------------"
        if not os.path.exists(('/').join((path,f))):
            os.makedirs(('/').join((path,f)))
        misc.imsave(('/').join((path,f,data.replace(".npz","_")+str(h)+'_'+str(w)+".png")),im)
    #misc.imshow(im3)



def main(argv):
 path = argv[0]
 h = int(argv[1])
 w = int(argv[2])


 for fold in tqdm(os.listdir(path)):
    for t in os.listdir(("/").join((path,fold))):
        for c in os.listdir(("/").join((path,fold,t))):
            for file in os.listdir(("/").join((path,fold,t,c))):
                #print ("/").join((path,fold,t,c,file))
                if os.path.isfile(("/").join((path,fold,t,c,file))):
                    #print ("/").join((path,fold,t,c)),"+++++++++"
                    Convert2Image(("/").join((path,fold,t,c)),file,h,w)








if __name__ == '__main__':
 main(sys.argv[1:])
