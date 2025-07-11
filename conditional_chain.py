from langchain_huggingface import ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


# Load environment variables
load_dotenv()

# Define the Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Create the chat model
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


# Define Pydantic Model for Feedback
class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

# 1st prompt - Sentiment Classification
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

# Prompts for responses
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
)

# Branching logic (Fixed `sentiment` access and typo)
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Couldn't determine sentiment"),
)

# Full chain
chain = classifier_chain | branch_chain

# Test with a sample input
result = chain.invoke({"feedback": "This is a nice smartphone. I love it."})
print(result)
