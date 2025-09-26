from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_aws import ChatBedrock as Bedrock

chat = Bedrock(
     region_name = "us-east-1",
     model_kwargs={"temperature":1,"top_k":250,"top_p":0.999,"anthropic_version":"bedrock-2023-05-31"},
     model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

multi_var_prompt = PromptTemplate(
     input_variables=["company"],
     template="Create a list with the names of the main metrics tracked in the reports of {company}?",
)
 
chain = LLMChain(llm=chat, prompt=multi_var_prompt)
answers = chain.invoke("Vanguard")
print(answers)

answers = chain.invoke("AWS")
print(answers)