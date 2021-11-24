import cv2
import numpy as np
import os 
import pywt
class wavelet:
    def __init__(self,baseDir) -> None:
        self.baseDir = baseDir
        self.X = []
        self.Y = []
        labels = os.listdir(os.path.join(baseDir,'processed'))
        self.dar = {j:i for i,j in enumerate(labels)}



    def __pathfile(self):
        for label in os.listdir(os.path.join(self.baseDir,'processed')):
            for imgName in os.listdir(os.path.join(self.baseDir,'processed',label)):
                imgPath = os.path.join(self.baseDir,"processed",label,imgName)
                yield imgPath           

    def w2d(self,img, mode='haar', level=1):
        imArray = img
        #Datatype conversions
        #convert to grayscale
        imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
        #convert to float
        imArray =  np.float32(imArray)   
        imArray /= 255
        # compute coefficients 
        coeffs=pywt.wavedec2(imArray, mode, level=level)

        #Process Coefficients
        coeffs_H=list(coeffs)  
        coeffs_H[0] *= 0;  

        # reconstruction
        imArray_H= pywt.waverec2(coeffs_H, mode)
        imArray_H *= 255
        imArray_H =  np.uint8(imArray_H)

        return imArray_H
    
    def wavelet(self):
        for path in self.__pathfile():
            # print(path)
            img = cv2.imread(path)
            wave = self.w2d(img,'db1',5)
            img_rs = cv2.resize(img,(32,32))
            wave_rs = cv2.resize(wave,(32,32))
            combined_img = np.vstack((img_rs.reshape(32*32*3,1),wave_rs.reshape(32*32,1)))
            self.X.append(combined_img)
            self.Y.append(self.dar[os.path.basename(os.path.dirname(path))])
        self.X = np.array(self.X).reshape(len(self.X),4096).astype(float)


# wav = wavelet(os.path.abspath(os.getcwd()))

# wav.wavelet()
# print(wav.X.shape)
# print(wav.Y)
