import argparse
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


DEFAULT_MODEL = "llama2:7b-chat"
DEFAULT_EMBEDDING_MODEL = "llama2:7b-chat"
DEFAULT_PATH = "research"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_NUM_RETRIEVED_DOCS = 9
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"  


CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to make it stand alone.

Conversation History:
{chat_history}

Follow-Up Question: {question}
Rephrased Question:"""

ANSWER_TEMPLATE = """
### Instruction:
Using the context provided below, answer the question at the end. If you don’t know the answer, just say “I don’t know” and avoid making up an answer. Be as concise as possible.

## Context:
{context}

## Question:
{question}
"""

DOCUMENT_TEMPLATE = "{page_content}"

EVALUATION_PROMPT_TEMPLATE = """
### Instruction:
Using the context provided below, answer the question at the end with a step-by-step breakdown of the calculation. The output should follow this format:
[
"operation(",
"value1",
"value2",
")",
"EOF"
]

For example:
subtract("5829", "5735")
EOF

If you don’t know the answer, just say “I don’t know” and avoid making up an answer.

## Context:
{context}

## Question:
{question}

## Answer (step-by-step calculation):
"""


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(DOCUMENT_TEMPLATE)
EVALUATION_PROMPT = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)

def parse_arguments():
    """
    Parse command-line arguments and return the results.
    """
    parser = argparse.ArgumentParser(description="Run local LLM with RAG using Ollama.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="The name of the LLM model to use.")
    parser.add_argument("-e", "--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="The name of the embedding model to use.")
    parser.add_argument("-p", "--path", default=DEFAULT_PATH, help="The directory path containing documents to load.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="The size of text chunks for splitting.")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="The overlap size for text chunks.")
    parser.add_argument("--openai_key", default=DEFAULT_OPENAI_API_KEY, help="OpenAI API key for embedding and generation.")
    parser.add_argument("--openai_base", default=DEFAULT_OPENAI_API_BASE, help="OpenAI API base URL for embedding and generation.")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode to generate prediction files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # Add further processing code here
