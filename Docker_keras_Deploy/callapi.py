import json

import requests,numpy as np

def local():
    # url="http://127.0.0.1:5000/invocations"
    # url="http://localhost:8080/invocations"
    url="https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/crypto-infer1/invocations"
    arr = np.random.random((2, 1, 100, 121))
    data = { 'arr': arr.tolist()}
    response = requests.post(url, json=data)
    print(response.json())

def SM():
    import boto3
    endpoint_name = 'crypto-infer1'
    client = boto3.client('sagemaker-runtime')
    arr = np.random.random((2, 1, 100, 121))
    payload = json.dumps({ 'arr': arr.tolist()})
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload
    )
    print(response["Body"].read())

SM()