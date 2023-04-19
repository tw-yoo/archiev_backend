import io
import json

from codecarbon import EmissionsTracker
from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS, cross_origin

from emission_result import EmissionResult, LabelAndProb
from inference import get_inference_result_probs
import uuid
import os
import random

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

    # res = get_response(model_name, uid, inference_result)

    # os.remove(f"{uid}.csv")

    return model_name


@app.route("/test/inference/<model_name>", methods=['POST'])
@cross_origin(supports_credentials=True)
def test(model_name):
    return EmissionResult(
        model_name,
        step1_emission=0.1 * random.randint(1, 10),
        step2_emission=0.1 * random.randint(1, 10),
        step3_emission=0.1 * random.randint(1, 10),
        step4_emission=0.1 * random.randint(1, 10),
        step1_time=0.1 * random.randint(1, 10),
        step2_time=0.1 * random.randint(1, 10),
        step3_time=0.1 * random.randint(1, 10),
        step4_time=0.1 * random.randint(1, 10),
        inference_result=[
            LabelAndProb(
                f"test{i}", 0.1* random.randint(1, 10)
            ).__dict__ for i in range(10)
        ],
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
