# personality-classifier
A website running of flask server, a personality classifier that can detect the faces of four celebrities. 

## This project has three components.
- **Model**:- It contains the python machine learning model based on a logistic classifier. The model was working with a 90% classifier.
  - preprocess.py:- It takes in input a directory of image dataset, and will return the folder with faces cropped if a model is able to detect 2 eyes.
  - wavlet.py:- It takes an input of file from preprocess.py and vertically stack the wavelet transformation, so that model can classify easily.
  - face_detector.py:- In this jupyter file, the various method was tested and hyparameters were tuned. In the end, the best model was saved as saved_model.pkl
 
 - **Server**:- In this Flask server was made so that when the user uploads the image, the image gets processed and the machine learning model can predict the model.
  - process.py :- Takes input an image and convert it into a format as expected from the model.
  - util.py :- Takes input from process.py and outputs its predictions and handling base64 file
  - server.py :- Host the server and provide backend for the website.
  
 - **UI**:- Contains the UI for website,contains the HTML,CSS,js files. Data is communicated to the flask server and the result is presented on site.

## Screenshot
![Screenshot-1](https://github.com/simratsingh14/personality-classifier/blob/master/Screenshot/webpage1.jpg)
![Screenshot-2](https://github.com/simratsingh14/personality-classifier/blob/master/Screenshot/webpage2.jpg)

 
 
 
 **Special Thanks**
 Thank you [codebasics](https://www.youtube.com/playlist?list=PLeo1K3hjS3uvaRHZLl-jLovIjBP14QTXc) for making the playlist.
