import flask
import cv2
import matplotlib as plt
from tensorflow import keras
import numpy as np
import json
from PIL import Image
from io import BytesIO
from firebase_admin import credentials, firestore, initialize_app, storage
from datetime import datetime

app = flask.Flask(__name__)

cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {'storageBucket': 'safedrive-12f49.appspot.com'})
db = firestore.client()
classifications = db.collection('classifications')

model = keras.models.load_model('distractedDrivingModel')

category_map = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}

@app.route("/")
def home():
    return "safedrive home page"

@app.route("/classify-image", methods=["POST"])
def classifyImage():
    image = flask.request.files['image']
    image = image.read()

    image = np.fromstring(image,np.uint16)

    image = cv2.imdecode(image,cv2.IMREAD_GRAYSCALE)

    num = open("num.txt", "r").read()
    cv2.imwrite(f"driver_images/driver_image({num}).png", image)


    fileName = f"driver_images/driver_image({num}).png"
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
    blob.make_public()


    image_shaped = cv2.resize(image,(64, 64))

    new_img = image_shaped.reshape(-1, 64, 64, 1)

    y_prediction = model.predict(new_img, batch_size=40, verbose=1)

    prediction = category_map.get('c{}'.format(np.argmax(y_prediction)))


    the_date = datetime.now().strftime("%m/%d/%Y") 
    the_time = datetime.now().strftime("%H:%M:%S")

    data = {
        "date" : the_date,
        "time" : the_time,
        "image" : blob.public_url,
        "classification" : prediction,
    }

    with open("num.txt",'w') as file:
        file.write(f"{int(num) + 1}")

    classifications.add(data)

    return json.dumps(data)



if __name__ == '__main__':
    app.run(host="192.168.1.38", port=8000)