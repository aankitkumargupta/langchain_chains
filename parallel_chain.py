from langchain_huggingface import ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


# Load environment variables
load_dotenv()

# Define the Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Create the chat model
model1 = ChatHuggingFace(llm=llm)

model2 = ChatHuggingFace(llm=llm)

# model2 = ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

# 1st prompt - detailed report
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variable=["text"],
)

prompt2 = PromptTemplate(
    template="generate a 5 simple question and answerfrom the following \n  {text}",
    input_variable=["text"],
)

prompt3 = PromptTemplate(
    template="merge the provided notes and quiz into a single document \n  notes -> {notes} quiz -> {quiz}",
    input_variable=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser, "quiz": prompt2 | model2 | parser}
)

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain

text = """
Logistic Regression is a classification algorithm used to predict the probability of a binary outcome (0 or 1). It applies the sigmoid function to a linear combination of input features to produce a probability between 0 and 1. If the probability exceeds a threshold (usually 0.5), the output is classified as 1; otherwise, it is classified as 0.  

The sigmoid function is defined as:  

σ(z) = 1 / (1 + e^(-z))  

where z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b.  

The cost function used is the **binary cross-entropy (log-loss)**:  

J(θ) = - (1/m) Σ [ yᵢ log(h(xᵢ)) + (1 - yᵢ) log(1 - h(xᵢ)) ]  

Optimization is done using **gradient descent**, where weights are updated as:  

wⱼ := wⱼ - α (∂J/∂wⱼ)  

Variants include **binary logistic regression**, **multinomial logistic regression**, and **ordinal logistic regression**. Regularization techniques like **L1 (Lasso) and L2 (Ridge)** are used to prevent overfitting.  

Performance metrics include **accuracy, precision, recall, F1-score, ROC curve, and AUC score**. It works well for linearly separable data but struggles with complex patterns, requiring alternative models like decision trees or neural networks.
"""
result = chain.invoke({"text": text})

print(result)

# to visualise chain
chain.get_graph().print_ascii()
