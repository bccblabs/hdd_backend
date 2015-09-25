CAFFE_ROOT = "/home/ubuntu/caffe/"
# CAFFE_ROOT = '/Users/bski/Project/caffe/'
import sys
sys.path.insert (0, CAFFE_ROOT + "python")
import caffe, numpy as np

<<<<<<< HEAD
IMAGE_ROOT = "/tmp/"
MODELS_ROOT = "/home/ubuntu/caffe/models/cars/"
=======
>>>>>>> b01b0ead320d9990b9c770631bfbff54be1685c0
from flask import Flask, request, Response, jsonify
from mongokit import Connection, Document
from StringIO import StringIO
from werkzeug import secure_filename
import urllib, datetime, requests, traceback, os

MONGO_HOST = "localhost"
MONGO_PORT = 27017
IMAGE_ROOT = "/tmp/hdd_images/"
MODELS_ROOT = CAFFE_ROOT + "models/hdd_13_ft/"
ALLOWED_EXTENSIONS = set (['jpg', 'JPG'])

app = Flask (__name__)
app.config.from_object (__name__)
app.config ['UPLOAD_FOLDER'] = IMAGE_ROOT
app.config ['MONGO_HOST'] = MONGO_HOST
app.config ['MONGO_PORT'] = MONGO_PORT
app.config ['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
connection = Connection (app.config['MONGO_HOST'], app.config['MONGO_PORT'])

<<<<<<< HEAD
caffe.set_mode_cpu()
#blob = caffe.proto.caffe_pb2.BlobProto()
#data = open (MODELS_ROOT + "hdd_mean.binaryproto").read()
#blob.ParseFromString(data)
#mean_rs = np.array( caffe.io.blobproto_to_array(blob) )[0].mean(1).mean(1)
hdd_labels_file = MODELS_ROOT + "labels_450.txt"
=======
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)
hdd_labels_file = MODELS_ROOT + "labels.txt"
>>>>>>> b01b0ead320d9990b9c770631bfbff54be1685c0
labels = None
num_outs = 0
try:
    labels = np.loadtxt (hdd_labels_file, str, delimiter="\t")
    num_outs = len (labels)
except:
    app.logger.debug("[classifier] fuck, cant load label file")

c0 = caffe.Classifier (
<<<<<<< HEAD
            MODELS_ROOT + "deploy_450.prototxt",
            MODELS_ROOT + "450_45k.caffemodel",
=======
            MODELS_ROOT + "deploy.prototxt",
            MODELS_ROOT + "hdd_13_m0.caffemodel",
>>>>>>> b01b0ead320d9990b9c770631bfbff54be1685c0
            channel_swap = (2,1,0),
            raw_scale = 255,
            image_dims = (256, 256)
)
classifiers = [c0]


def classify_image (image_file_name):
    image = caffe.io.load_image(image_file_name)
    resized_image = caffe.io.resize_image (image, (256,256,3))
    res = np.zeros (num_outs * len (classifiers)).reshape (num_outs, len(classifiers))
    for i, x in enumerate (classifiers):
        res[:,i] = x.predict ([resized_image])[0]
    avg_probs = np.average (res, axis=1)
    top_k_idx = avg_probs.argsort()[-1:-6:-1]
    class_res = connection.Classification()
    class_res['image_path'] = image_file_name
    class_res['date_created'] = datetime.datetime.now()
    class_res['top_5'] = []
    for x in top_k_idx.tolist():
        res_dict = {}
        res_dict["class_name"] = labels.tolist()[x][0]
        res_dict["prob"] = avg_probs.tolist()[x]
        app.logger.debug (str(res_dict) + " " + image_file_name)
        class_res['top_5'].append (res_dict)
    class_res['top_1'] = class_res['top_5'][0]
    class_res.save()
    print class_res
    return class_res

@connection.register
class Classification (Document):
    __collection__ = "classifications"
    __database__   = "cars_450"
    use_dot_notation = True
    skip_validation = True
    structure = {
        "image_path": unicode,
        "date_created": datetime.datetime,
        "top_1": {
                "class_name": unicode,
                "prob": float
        },
        "top_3": [
            {
                "class_name": unicode,
                "prob": float
            }
        ]
    }
    required_fields = [
        "image_path",
        "date_created",
        "top_1",
        "top_3"
    ]
<<<<<<< HEAD
=======



@app.route("/hdd_classify", methods=['POST'])
def hdd_classify ():
    try:
        image_file = request.files['file']
        if image_file.filename.rsplit ('.', 1)[1] in ALLOWED_EXTENSIONS and '.' in image_file.filename:
            sec_fname = secure_filename (image_file.filename)
            image_full_path = os.path.join (app.config['UPLOAD_FOLDER'], sec_fname)
            image_file.save (image_full_path)
            classifier_resp = classify_image (image_full_path)
            response = Response (response = classifier_resp.to_json(), status=201, mimetype="application/json")
            return response
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        app.logger.error (" ".join(traceback.format_tb (exc_traceback)))
        resp = jsonify ({"msg": "Server Error"})
        resp.status_code = 500
        return resp


>>>>>>> b01b0ead320d9990b9c770631bfbff54be1685c0
@app.route("/classify")
def classify():
    try:
        num_outs = 14
        image_url = request.args.get ("image_url")
        image_file_name = IMAGE_ROOT + image_url.split("/")[-1]
        urllib.urlretrieve (image_url, image_file_name)
        app.logger.debug ("[classifier] image loaded to " + image_file_name)

        image = caffe.io.load_image(image_file_name)
        resized_image = caffe.io.resize_image (image, (256,256,3))
        res = np.zeros (num_outs * len (classifiers)).reshape (num_outs, len(classifiers))
        for i, x in enumerate (classifiers):
            res[:,i] = x.predict ([resized_image])[0]
        avg_probs = np.average (res, axis=1)
        top_k_idx = avg_probs.argsort()[-1:-4:-1]
        class_res = connection.Classification()
        class_res['image_url'] = image_url
        class_res['date_created'] = datetime.datetime.now()
        class_res['top_3'] = []
        for x in top_k_idx.tolist():
            res_dict = {}
            res_dict["class_name"] = labels.tolist()[x][0]
            res_dict["prob"] = avg_probs.tolist()[x]
            app.logger.debug (str(res_dict) + " " + image_file_name)
            class_res['top_3'].append (res_dict)
        class_res['top_1'] = class_res['top_3'][0]
        class_res.save()
        app.logger.debug ("[classifier] classification result saved." + str(class_res.to_json()))
        response = Response (response = class_res.to_json(), status=200, mimetype="application/json")
        return response
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        app.logger.error (" ".join(traceback.format_tb (exc_traceback)))
        resp = jsonify ({"msg": "Server Error"})
        resp.status_code = 500
        return resp

if __name__ == "__main__":
    if not app.debug:
        import logging
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler ("classifier_server.log", maxBytes=1024 * 20, backupCount=20)
        handler.setFormatter (logging.Formatter (
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
        handler.setLevel (logging.WARNING)
        app.logger.addHandler (handler)
    app.run(host="0.0.0.0")

