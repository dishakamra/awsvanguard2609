#Create a service client by name using the default session.
import json
import os
import sys

import boto3
import botocore

module_path = ".."
sys.path.append(os.path.abspath(module_path))

bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_DEFAULT_REGION", None))

# create the prompt
prompt_data = """
Command: Write an email from Bob, Customer Service Manager, AnyCompany to the customer "John Doe" 
who provided negative feedback on the service provided by our customer support 
engineer"""

body = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": [{"text": prompt_data}]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 8192,
        "stopSequences": [],
        "temperature": 0,
        "topP": 0.9
    }
}) 

#invoke model
modelId = 'amazon.nova-lite-v1:0' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'
outputText = "\n"
try:

    response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    outputText = response_body.get('output').get('message').get('content')[0].get('text')

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# Print the complete response from Nova Lite
print(outputText)