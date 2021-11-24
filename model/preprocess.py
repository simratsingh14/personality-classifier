import cv2
import numpy as np
import os 


class Preprocessing:
    def __init__(self,pathDir,labelList) -> None:
        self.pathDir = pathDir
        self.labelList = labelList
    def __convert(self,path):
        img = cv2.imread(path)
        # print(path)
        if img is not None:
            # gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
    def __faceCordinate(self,img):
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        face_xml_path = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
        faceModel = cv2.CascadeClassifier(face_xml_path)
        faces = faceModel.detectMultiScale(img,1.3,5)
        return faces
    def __eyeCordinate(self,img):
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        eye_xml_path = os.path.join(cv2_base_dir,'data/haarcascade_eye.xml')
        eyeModel = cv2.CascadeClassifier(eye_xml_path)
        eyes = eyeModel.detectMultiScale(img,1.3,5)
        return eyes
    def __roi(self,img,faces):
        for (x,y,w,h) in faces:
            roi = img[y:y+h,x:x+w]
        return roi
            


    def __imgPath(self):
        for i in self.labelList:
            for img in os.listdir(os.path.join(self.pathDir,"data\\"+i)):
                path = os.path.join(os.path.join(self.pathDir,"data\\"+i),img)
                # print(path)
                yield path

    def __makeDirectory(self):
        # savingDirectory = os.makedirs(os.path.join(self.pathDir,"processed"),exist_ok=True)
        for label in self.labelList:
            os.makedirs(os.path.join(self.pathDir,"processed\\" + label),exist_ok=True)
        return

    def __saveDirectory(self,label,img,name):
        directory = os.path.join(self.pathDir,"processed\\"+label)
        print(os.path.join(directory,name))
        try:
            cv2.imwrite(os.path.join(directory,name),img)
        except:
            return
        return
        
    def preprocess(self):
        self.__makeDirectory()
        count = 0
        for path in self.__imgPath():
            img = self.__convert(path)
            if img is None:
                continue
            faces = self.__faceCordinate(img)
            eyes = self.__eyeCordinate(img)
            if len(faces) > 0 and len(eyes) > 1:
                count+=1
                label = os.path.basename(os.path.dirname(path))
                name = label + '_' + str(count) + '.jpg'
                # print(name)
                roi = self.__roi(img,faces)  # Returns Cropped image with faces
                self.__saveDirectory(label,roi,name) 

                                  

       
            



baseDir = os.path.abspath(os.getcwd())
label = os.listdir(os.path.join(baseDir,'data'))
P = Preprocessing(baseDir,labelList=label)

P.preprocess()









