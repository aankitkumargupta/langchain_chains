from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Define the Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Create the chat model
model = ChatHuggingFace(llm=llm)

# 1st prompt - detailed report
prompt1 = PromptTemplate(
    template="write a detailed report on {topic} in  sentences",
    input_variable=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a 5 pointer summary on the following text \n {text}",
    input_variable=["text"],
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser


result = chain.invoke({"topic": "basketball"})

print(result)

# to visualise chain
chain.get_graph().print_ascii()
