import boto3,sagemaker
from sagemaker import get_execution_role

# tar cvfz cryptodir.tar.gz --exclude=".ipynb*" dqn_trend-following-v0_cnn_weights_model_weights.h5  code/inference.py

# role = get_execution_role()
role="arn:aws:iam::793613833548:role/service-role/AmazonSageMaker-ExecutionRole-20221201T031659"
region = boto3.Session().region_name

sm_client = boto3.client(service_name="sagemaker",region_name=region)
runtime_sm_client = boto3.client(service_name="sagemaker-runtime")

# account_id = boto3.client("sts").get_caller_identity()["Account"]
account_id="7dea65d97e3a"
region = boto3.Session().region_name

bucket = "sagemaker-{}-{}".format(region, account_id)
prefix = "kerascrypto-endpoint"

sess = boto3.Session()
sm= sess.client('sagemaker',region_name=region)
print(account_id,region,role)

# imageurl="793613833548.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tf-crpto-example"
imageurl="793613833548.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tf-keras-example"
modelurl="s3://sagemaker-us-west-2-793613833548/model/kerasdir.tar.gz"

# https://docs.aws.amazon.com/sagemaker/latest/dg/neo-deployment-hosting-services-boto3.html
create_model_api_response = sm_client.create_model(
                                    ModelName='crypto-model1',
                                    PrimaryContainer={
                                        'Image': imageurl,
                                        # 'ModelDataUrl': modelurl,
                                        'Environment': {
                                        }
                                    },
                                    ExecutionRoleArn=role
                            )
print(create_model_api_response)
# create sagemaker endpoint config
create_endpoint_config_api_response = sm_client.create_endpoint_config(
                                            EndpointConfigName="cryptoConfig1",
                                            ProductionVariants=[
                                                {
                                                    'VariantName': "crypto-model",
                                                    'ModelName': 'crypto-model1',
                                                    'InitialInstanceCount': 1,
                                                    'InstanceType': 'ml.t2.medium'
                                                },
                                            ]
                                       )

print ("create_endpoint_config API response", create_endpoint_config_api_response)


# create sagemaker endpoint
create_endpoint_api_response = sm_client.create_endpoint(
                                    EndpointName='crypto-infer1',
                                    EndpointConfigName='cryptoConfig1',
                                )

print ("create_endpoint API response", create_endpoint_api_response)