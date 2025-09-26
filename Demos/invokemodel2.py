import boto3
import json
bedrock_rt = boto3.client(service_name='bedrock-runtime')
prompt = "Write an essay for living on Mars using 10 sentences."

configs= {
     "inputText": prompt,
     "textGenerationConfig": {
          "temperature":0
     }
}

body=json.dumps(configs)

accept = 'application/json'
contentType = 'application/json'
modelId = 'amazon.titan-tg1-large'

response = bedrock_rt.invoke_model_with_response_stream(
     modelId=modelId,
     body=body,
     accept=accept,
     contentType=contentType
)

stream = response.get('body')
if stream:
     for event in stream:
          chunk = event.get('chunk')
          if chunk:
               print((json.loads(chunk.get('bytes').decode())))