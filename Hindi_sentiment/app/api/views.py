from flask import Blueprint, jsonify
import urllib.request
from flask import Flask, request, redirect
from werkzeug.utils import secure_filename
import urllib.request
import time
import os
import json
from app import app
import pickle
import unittest
from flask_httpauth import HTTPBasicAuth
from app.hindi_sentiment_nlu_main_tensorflow_serving import predict_class
from app.hindi_sentiment_nlu_main_tensorflow_serving import tensorflow_serve
blueprint_agent = Blueprint("agent", __name__)

@app.route("/predict-text", methods=["POST"])
def predict_result() -> dict:
    
    if "text[]" not in request.form:
        resp = jsonify({"message": "No file part in the request"})
        resp.status_code = 400
        return resp

    text = request.form.get("text[]")
	

    errors = {}
    success = False
    file_names = []
    c = 0
    
       
    if len(str(text))!=0 :
            success = True
    else:
            errors[text] = "Text is empty"

    if success:
          resp = jsonify({"message": "Text successfully uploaded.."})
          resp.status_code = 201
          with open('inputs/word2id_100.pkl', 'rb') as handle:
              word2id = pickle.load(handle)

          with open('inputs/word_vec_100.pkl','rb') as f:
              word_vec = pickle.load(f)
          print('Files are loaded')
          data=text
          print('the input data is ',data)
          tensorflow_serve()
          final_dict=predict_class(data,word2id,word_vec)
          return jsonify({'data': final_dict})
   
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


if __name__ == "__main__":
    app.run()