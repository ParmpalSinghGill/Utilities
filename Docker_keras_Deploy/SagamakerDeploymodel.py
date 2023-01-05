import boto3,sagemaker
from sagemaker import get_execution_role

def deleteAllEndpoint(sm_client):
    response = sm_client.list_endpoint_configs()
    for cnf in response["EndpointConfigs"]:
        print("deleting",cnf)
        response = sm_client.delete_endpoint_config(EndpointConfigName=cnf["EndpointConfigName"])
        print(response)
    response = sm_client.list_models()
    for cnf in response["Models"]:
        print("deleting",cnf)
        response = sm_client.delete_model(ModelName=cnf["ModelName"])
        print(response)

# tar cvfz cryptodir.tar.gz --exclude=".ipynb*" dqn_trend-following-v0_cnn_weights_model_weights.h5  code/inference.py

# role = get_execution_role() # this role have Full access of Sagemaker and S3 
role="arn:aws:iam::793613833548:role/service-role/AmazonSageMaker-ExecutionRole-20221201T031659"  
region = boto3.Session().region_name

sm_client = boto3.client(service_name="sagemaker",region_name=region)
runtime_sm_client = boto3.client(service_name="sagemaker-runtime")

# account_id = boto3.client("sts").get_caller_identity()["Account"]
account_id="793613833548"
region = boto3.Session().region_name

bucket = "sagemaker-{}-{}".format(region, account_id)
prefix = "kerascrypto-endpoint"

sess = boto3.Session()
sm= sess.client('sagemaker',region_name=region)
print(account_id,region,role)


def deployEndpoint(sm_client,ModelName,EndpointConfigName,EndpointName):
    # https://docs.aws.amazon.com/sagemaker/latest/dg/neo-deployment-hosting-services-boto3.html
    create_model_api_response = sm_client.create_model(
                                        ModelName=ModelName,
                                        PrimaryContainer={
                                            'Image': imageurl,'ModelDataUrl': modelurl,
                                            'Environment': {}
                                        },
                                        ExecutionRoleArn=role
                                )
    print(create_model_api_response)
    # create sagemaker endpoint config
    create_endpoint_config_api_response = sm_client.create_endpoint_config(
                                                EndpointConfigName=EndpointConfigName,
                                                ProductionVariants=[
                                                    {
                                                        'VariantName': "keras-model",
                                                        'ModelName':ModelName,
                                                        'InitialInstanceCount': 1,
                                                        'InstanceType': 'ml.t2.medium'
                                                    },
                                                ]
                                           )

    print ("create_endpoint_config API response", create_endpoint_config_api_response)


    # create sagemaker endpoint
    create_endpoint_api_response = sm_client.create_endpoint(
                                        EndpointName=EndpointName,
                                        EndpointConfigName=EndpointConfigName,
                                    )

    print ("create_endpoint API response", create_endpoint_api_response)

def deployEndpointServerless(sm_client,ModelName,EndpointConfigName,EndpointName):
    # https://docs.aws.amazon.com/sagemaker/latest/dg/neo-deployment-hosting-services-boto3.html
    create_model_api_response = sm_client.create_model(
                                        ModelName=ModelName,
                                        PrimaryContainer={
                                            'Image': imageurl,'ModelDataUrl': modelurl,
                                            'Environment': {}
                                        },
                                        ExecutionRoleArn=role
                                )
    print(create_model_api_response)
    # create sagemaker endpoint config
    create_endpoint_config_api_response = sm_client.create_endpoint_config(
                                                EndpointConfigName=EndpointConfigName,
                                                ProductionVariants=[
                                                    {
                                                        'VariantName': "keras-model",
                                                        'ModelName':ModelName,
                                                        "ServerlessConfig": {
                                                            "MemorySizeInMB": 4096,
                                                            "MaxConcurrency": 1,
                                                        },
                                                    },
                                                ]
                                           )

    print ("create_endpoint_config Severless API response", create_endpoint_config_api_response)


    # create sagemaker endpoint
    create_endpoint_api_response = sm_client.create_endpoint(
                                        EndpointName=EndpointName,
                                        EndpointConfigName=EndpointConfigName,
                                    )

    print ("create_endpoint API Severless response", create_endpoint_api_response)



def deleteEndpoint(sm_client,ModelName,EndpointConfigName,EndpointName):
    try:
        response = sm_client.delete_endpoint(
            EndpointName=EndpointName
        )
        print("Deleted Endpoint",response)
    except: pass
    try:
        response = sm_client.delete_endpoint_config(
            EndpointConfigName=EndpointConfigName
        )
        print("Deleted EndpointConfig", response)
    except: pass
    try:
        response = sm_client.delete_model(
            ModelName=ModelName
        )
        print("Deleted Model", response)
    except: pass

def deployModel():
    model = sagemaker.model.Model(image_uri=imageurl,model_data = modelurl,role = role)
    print(model)
    resp=model.deploy(initial_instance_count=1,instance_type='ml.t2.medium')
    print(resp)

# imageurl="793613833548.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tf-crpto-example"
# imageurl="793613833548.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tf-keras-example"
imageurl="793613833548.dkr.ecr.us-west-2.amazonaws.com/tf-keras-image:latest"
# modelurl="s3://sagemaker-us-west-2-793613833548/model/kerasdir.tar.gz"
# modelurl="s3://sagemaker-us-west-2-793613833548/model/keras_model_weight.tar.gz"
modelurl="s3://sagemaker-us-west-2-793613833548/model/keras_model_weight.tar.gz"


# EndpointName='Keras-Endpoint'
# ModelName='keras-test-model'
# EndpointConfigName="kf-config"

version=""
# version="-2"
EndpointName='Keras-Endpoint'+version
ModelName='keras-test-model'+version
EndpointConfigName="kf-config"+version

# deployEndpoint(sm_client,ModelName,EndpointConfigName,EndpointName)
# deployEndpointServerless(sm_client,ModelName,EndpointConfigName,EndpointName)
# deleteEndpoint(sm_client,ModelName,EndpointConfigName,EndpointName)
# deleteAllEndpoint(sm_client)
# deployModel()



