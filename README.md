# LangChain Demo: Chain Types Using HuggingFace Models

This repository contains four examples demonstrating how to build **LangChain chains** with Hugging Face language models like [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it). Each script introduces a different chaining strategy—**simple**, **sequential**, **parallel**, and **conditional**.

These patterns are useful when building LLM-powered tools that need structured responses, branching logic, or modular pipelines.

---

## 🧰 Setup Instructions

### 1. Install Python packages

```bash
pip install langchain langchain-huggingface python-dotenv pydantic
```

(For optional Anthropic support)

```bash
pip install langchain-anthropic
```

### 2. Set up environment variables

Create a `.env` file in the root directory:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
# Optional if using Claude
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## 🔗 Project Overview

| File                   | Chain Type  | Description                                                                 |
| ---------------------- | ----------- | --------------------------------------------------------------------------- |
| `simple_chain.py`      | Simple      | One-step generation using prompt + model + output parser.                   |
| `sequential_chain.py`  | Sequential  | Multi-step pipeline where each step depends on the previous output.         |
| `parallel_chain.py`    | Parallel    | Executes two tasks (notes and quiz) in parallel, then merges results.       |
| `conditional_chain.py` | Conditional | Executes logic-based routing (sentiment-based branching) for dynamic paths. |

---

## 📄 1. `simple_chain.py`

### ✅ What it does:

Generates a **short report** (5 sentences) on a given topic using a single prompt.

### 🔁 Chain Flow:

```
PromptTemplate (write about topic) → ChatHuggingFace → StrOutputParser
```

### 📥 Input:

```python
{"topic": "basketball"}
```

### 📤 Output:

A paragraph about basketball, e.g.:

> "Basketball is a fast-paced team sport that involves two teams of five players..."

---

## 📄 2. `sequential_chain.py`

### ✅ What it does:

Takes a topic, generates a **detailed report**, and then summarizes it into **5 bullet points**.

### 🔁 Chain Flow:

```
Prompt (detailed report) → Model → Output → Prompt (summary) → Model → StrOutputParser
```

### 📥 Input:

```python
{"topic": "basketball"}
```

### 📤 Output:

A 5-point summary of the generated report:

```
1. Basketball is a team sport...
2. It involves shooting a ball through a hoop...
...
```

---

## 📄 3. `parallel_chain.py`

### ✅ What it does:

Processes a given educational text to simultaneously generate:

* ✍️ A **short summary**
* ❓ A **quiz with 5 Q\&A pairs**
* 📄 Merges both into a final formatted output

### 🔁 Chain Flow:

```
          → (Prompt1 → Model1 → notes)
Text → RunnableParallel
          → (Prompt2 → Model2 → quiz)

Then:
notes + quiz → Prompt3 (merge) → Model → Output
```

### 📥 Input:

Long-form content, e.g., a passage on Logistic Regression

### 📤 Output:

```json
{
  "notes": "Logistic Regression is a classification algorithm...",
  "quiz": "Q1: What is logistic regression? A: It is...",
  "merged": "NOTES: ... QUIZ: ..."
}
```

---

## 📄 4. `conditional_chain.py`

### ✅ What it does:

Performs **sentiment classification** on feedback, and based on the result, gives a **custom response**.

### 🔁 Chain Flow:

```
Step 1: Classify sentiment → Pydantic Parser
Step 2: If Positive → Prompt → Model
        If Negative → Prompt → Model
        Else → Fallback message
```

### 📥 Input:

```python
{"feedback": "This is a nice smartphone. I love it."}
```

### 📤 Output:

```
"Thank you for your positive feedback! We're glad you're enjoying the smartphone."
```

If feedback is negative:

```
"We're sorry to hear that. We'll work to improve it."
```

---

## 🔍 Chain Visualization

Each script ends with:

```python
chain.get_graph().print_ascii()
```

This prints a visual ASCII diagram of how components are connected, helpful for understanding chain logic.

---

## 📌 Summary

This repo demonstrates how to:

* Structure prompt pipelines using LangChain
* Use Hugging Face models for multi-step reasoning
* Implement parallel and conditional workflows
* Parse outputs into structured formats (e.g., using `PydanticOutputParser`)

Whether you're building feedback systems, quiz generators, or LLM-powered summaries—these templates can be adapted easily.




