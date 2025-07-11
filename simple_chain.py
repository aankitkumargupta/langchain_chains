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
prompt = PromptTemplate(
    template="write a detailed report on {topic} in 5 sentences",
    input_variable=["topic"],
)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "basketball"})

print(result)

# to visualise chain
chain.get_graph().print_ascii()
