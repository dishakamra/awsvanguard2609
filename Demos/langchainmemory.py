import boto3
from langchain.chains import ConversationChain
from langchain_aws import BedrockLLM
from langchain.memory import ConversationBufferMemory
bedrock_client = boto3.client('bedrock-runtime',region_name="us-east-1")


titan_llm = BedrockLLM(model_id="amazon.titan-tg1-large", client=bedrock_client)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=titan_llm, verbose=True, memory=memory
)

print(conversation.predict(input="Hi! I am in Los Angeles. What are some of the popular sightseeing places?"))
print(conversation.predict(input="What is closest beach that I can go to?"))
print(conversation.predict(input="What is the weather like in Los Angeles in December?"))
