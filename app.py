import os
import base64
from io import BytesIO
from fastai import *
from fastai.vision import *
from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import BadRequest
#from hints import fact_finder

def evaluate_image(img) -> str:
    pred_class, pred_idx, outputs = trained_model.predict(img)
    return pred_class

def load_model():
    #path = '/root/JupyterNotebooks/roomba/livingroom2/'
    path = '/root/JupyterNotebooks/roomba/livingroom2/dataset_from_roomba'
    classes = ['FRONT_CRASH', 'LEFT_BUMP', 'RIGHT_BUMP', 'YES'] 
    data = ImageDataBunch.single_from_classes(path, classes, 0, size=800).normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet34)
    learn.load('stage-2')
    return learn

app = Flask(__name__)
app.config['DEBUG'] = False
trained_model = load_model()

@app.route('/', methods=['POST'])
def eval_image():
    #input_file = request.args.get('data')
    data = request.data
    input_file=base64.decodestring(data)
    #print(input_file)
    #r = request.get_json()
    #r.
    #base64.decodestring(image_64_encode) 
   # if not input_file:
   #     return BadRequest("File is not present in the request")
   # if input_file.filename == '':
   #     return BadRequest("Filename is not present in the request")
   # if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
   #     return BadRequest("Invalid file type")
    #input_file.save("test.jpg","JPEG")
    src =BytesIO(input_file) 
    print (src)
    #input_buffer = BytesIO()
    #print( input_buffer)
    #input_file.save(input_buffer)
    
    guess = evaluate_image(open_image(src))
    print (guess)
    #hint = fact_finder(guess)
    return str(guess)
    #return jsonify({
    # 'guess': guess
    #})

if __name__ == "__main__":
    app.run(host='192.168.10.7', threaded=False)
