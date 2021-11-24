import joblib
import json
import os

import numpy as np
from process import Preprocess

baseDir = os.path.join(os.path.abspath(os.getcwd()),"server")
__class_name_to_number = {}
__number_to_class_name = {}
__model = None


def classname2number(cname):
    global __class_name_to_number
    return __class_name_to_number[cname]

def number2classname(number):
    global __number_to_class_name
    return __number_to_class_name[number]

def load_saved_artifact():
    print("Loading saved artifact...........start")
    global __class_name_to_number
    global __number_to_class_name

    with open(baseDir + '\\artifact/class_dictionary.json',"r") as f:
        __class_name_to_number = json.load(f)
        __number_to_class_name = {v:k for k,v in __class_name_to_number.items()}

    global __model 
    if __model is None:
        with open(baseDir + '\\artifact/saved_model.pkl',"rb") as f:
            __model = joblib.load(f)
    print("loading saved artifact........done")

def classifier(image_base64_date,file_path=None):
    X = Preprocess(image_base64_date,file_path)
    result = []
    for x in X:
        result.append({
            'class':number2classname(__model.predict(x)[0]),
            'class_probability': np.round(__model.predict_proba(x)*100,2).tolist(),
            'class_dictionary': __class_name_to_number
            }) 
    return result

def get_test_image():
    with open(baseDir + "\\testimage"+ "\\testimage.txt") as f:
        return f.read()

if __name__ == "__main__":
    load_saved_artifact()
    print(classifier(get_test_image()))
    print(classifier(None,os.path.join(baseDir,"testimage","Jerry_Seinfeld_RT1.jpg")))
    print(classifier(None,os.path.join(baseDir,"testimage","16337782734782.jpg")))
    print(classifier(None,os.path.join(baseDir,"testimage","e8fbbfb415498152e3deef6d215b4249.jpg")))
