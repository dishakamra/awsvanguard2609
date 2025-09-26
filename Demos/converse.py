import boto3
client = boto3.client("bedrock-runtime")
#model_id = "amazon.titan-tg1-large"
#model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0" 
model_id = "amazon.nova-lite-v1:0"

# Inference parameters to use.
temperature = 0.1
top_p = 0.2

inference_config={"temperature": temperature,"topP": top_p}

# Setup the system prompts and messages to send to the model.
system_prompts = [{"text": "You are a helpful assistance. Please answer the query politely."}]
conversation = [
    {
        "role": "user",
        "content": [{"text": "Hello, what is the capital of France?"}]
    }
]
# Use converse API
response = client.converse(
    modelId=model_id,
    system=system_prompts,
    messages=conversation,
    inferenceConfig=inference_config
)
print(response['output']['message']["content"][0]["text"])