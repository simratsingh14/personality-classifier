import cv2
import numpy as np
import pywt
import base64
import os


def convert(bs64):
    encoded_date = bs64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_date),np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img
    



def faceCordinate(img):
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    face_xml_path = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    faceModel = cv2.CascadeClassifier(face_xml_path)
    faces = faceModel.detectMultiScale(img,1.3,5)
    return faces

def eyeCordinate(img):
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    eye_xml_path = os.path.join(cv2_base_dir,'data/haarcascade_eye.xml')
    eyeModel = cv2.CascadeClassifier(eye_xml_path)
    eyes = eyeModel.detectMultiScale(img,1.3,5)
    return eyes

def rol(img,faces):
    roi = []
    for (x,y,w,h) in faces:
        roi.append(img[y:y+h,x:x+w])
    return roi



def w2d(img, mode='haar', level=1):
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





def croppedFace(base64,file_path=None):
    if base64 is None:
        img = cv2.imread(file_path)
    else:
        img = convert(base64)
    # print(img,file_path)
    faces = faceCordinate(img)
    # print(faces)
    eyes = eyeCordinate(img)
    if len(faces) > 0 and len(eyes) > 1:
        roi = rol(img,faces)
        return roi
    else:
        return -1
def Preprocess(base64,file_path=None):
    face_imgs = croppedFace(base64,file_path)
    X = []
    try: 
        for face_img in face_imgs:
            wav = w2d(face_img,'db1',5)
            face_rs = cv2.resize(face_img,(32,32))
            wav_rs = cv2.resize(wav,(32,32))
            combined_img = np.vstack((face_rs.reshape(32*32*3,1),wav_rs.reshape(32*32,1)))
            X.append(combined_img.reshape(1,4096).astype(float))
        return X
    except:
        # print("Face not found")
        return []

    

                                     

if __name__ == "__main__":
    pass

            







