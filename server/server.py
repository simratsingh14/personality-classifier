from flask import Flask, request, jsonify
from util import classifier,load_saved_artifact

app = Flask(__name__)

@app.route('/classify_image',methods = ['GET','POST'])
def clessify_image():
    imageData = request.form['image_data']
    response = jsonify(classifier(imageData,None))
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


if __name__ == "__main__":
    print("Starting Personality Classifier")
    load_saved_artifact()
    app.run(port=5000)


