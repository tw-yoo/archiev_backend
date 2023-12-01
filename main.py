import io
import json

from codecarbon import EmissionsTracker
from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS, cross_origin

import torch
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import certifi
import urllib
import torchvision.transforms as transforms

from emission_result import EmissionResult, LabelAndProb
from inference import get_inference_result_probs
import uuid
import os
import random

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

model_df = pd.read_csv("model.csv")

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/model_list")
@cross_origin(supports_credentials=True)
def get_model_list():

    model_list = [
        r['name'] for i, r in model_df.iterrows()
    ]

    return model_list


@app.route("/inference/<model_name>", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_inference_result(model_name: str):

    # 모델 경로 가져오기
    model_url = get_model_url(model_name)
    req = request.data

    # 모델마다 id 만들기
    uid = str(uuid.uuid1())

    inference_result = get_inference_result_probs(uid,model_df, model_name, req)

    res = get_response(model_name, uid, inference_result)

    os.remove(f"{uid}.csv")

    return res.__dict__


@app.route("/test/inference/<model_name>", methods=['POST'])
@cross_origin(supports_credentials=True)
def test(model_name: str):

    if model_name == "AlexNet":
        model = models.alexnet(weights=True)
    elif model_name == "DenseNet":
        model = models.densenet121(weights=True)
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(weights=True)
    elif model_name == "GoogleNet":
        model = models.googlenet(weights=True)
    elif model_name == "InceptionV3":
        model = models.inception_v3(weights=True)
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=True)
    elif model_name == "ResNet":
         model = models.resnet50(weights=True)
    elif model_name == "ShuffleNet":
        model = models.shufflenet_v2_x1_0(weights=True)
    else:
        model = models.vgg11(weights=True)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_io = io.BytesIO(request.data)
    image = Image.open(data_io)

    img = image.convert("RGB")
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():  # No need to calculate gradients
        output = model(batch_t)

    url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 10)

    inference_result = []
    for i in range(top5_prob.size(0)):
        inference_result.append(
            LabelAndProb(
                categories[top5_catid[i]],
                top5_prob[i].item()
            ).__dict__
        )
        print(categories[top5_catid[i]], top5_prob[i].item())

    return EmissionResult(
        model_name,
        step1_emission=0.1 * random.randint(1, 10) * 0.2,
        step2_emission=0.1 * random.randint(1, 10) * 0.4,
        step3_emission=0.1 * random.randint(1, 10) * 1.5,
        step4_emission=0.1 * random.randint(1, 10) * 1.5,
        step1_time=0.1 * random.randint(1, 10) * 0.2,
        step2_time=0.1 * random.randint(1, 10) * 0.4,
        step3_time=0.1 * random.randint(1, 10) * 1.5,
        step4_time=0.1 * random.randint(1, 10) * 1.5,
        inference_result=inference_result
        # inference_result=[
        #     LabelAndProb(
        #         f"test{i}", 0.1* random.randint(1, 10)
        #     ).__dict__ for i in range(10)
        # ],
    ).__dict__


def get_response(model_name, uid, inference_result):
    emission_df = pd.read_csv(f"{uid}.csv")

    value = 1000

    return EmissionResult(
        model_name,
        step1_emission=emission_df.query("project_name == 'step1'").iloc[0]['emissions'] * value,
        step2_emission=emission_df.query("project_name == 'step2'").iloc[0]['emissions'] * value,
        step3_emission=emission_df.query("project_name == 'step3'").iloc[0]['emissions'] * value,
        step4_emission=emission_df.query("project_name == 'step4'").iloc[0]['emissions'] * value,
        step1_time=emission_df.query("project_name == 'step1'").iloc[0]['duration'],
        step2_time=emission_df.query("project_name == 'step2'").iloc[0]['duration'],
        step3_time=emission_df.query("project_name == 'step3'").iloc[0]['duration'],
        step4_time=emission_df.query("project_name == 'step4'").iloc[0]['duration'],
        inference_result=inference_result,
    )


def get_model_url(model_name):
    query_result = model_df.query(f"name == '{model_name}'").iloc[0]
    return query_result['url']


@app.route("/download-models-1", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_1():
    models.alexnet(weights=True)

@app.route("/download-models-2", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_2():
    models.densenet121(weights=True)

@app.route("/download-models-3", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_3():
    models.efficientnet_b0(weights=True)

@app.route("/download-models-4", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_4():
    models.googlenet(weights=True)

@app.route("/download-models-5", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_5():
    models.inception_v3(weights=True)

@app.route("/download-models-6", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_6():
    models.mobilenet_v2(weights=True)

@app.route("/download-models-7", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_7():
    models.resnet50(weights=True)

@app.route("/download-models-8", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_8():
    models.shufflenet_v2_x1_0(weights=True)

@app.route("/download-models-9", methods=['POST'])
@cross_origin(supports_credentials=True)
def download_pretrained_models_9():
    models.vgg11(weights=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

