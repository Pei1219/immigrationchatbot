from operator import itemgetter
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from default_config import DEFAULT_NUM_RETRIEVED_DOCS, CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT, EVALUATION_PROMPT
from query_rewrite import QueryRewriter
from reranker import Reranker
import torch
from config import Config
from typing import List, Dict
from evaluation.evaluate_metrics import calculate_bleu, calculate_rouge
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


config = Config()  
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化 Reranker 和 QueryRewriter
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
reranker = Reranker(config, device)
query_rewriter = QueryRewriter(config.llm_name, "conversation_id", memory, reranker)

def get_streaming_chain(question: str, memory, llm, db):
    logging.info("Initializing streaming chain")
    retriever = db.as_retriever(search_kwargs={"k": DEFAULT_NUM_RETRIEVED_DOCS})
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(lambda x: "\n".join(
            [f"{item['role']}: {item['content']}" for item in x["memory"]]
        )),
    )


    rewritten_question = query_rewriter.rewrite(question)
    logging.info(f"Rewritten Question: {rewritten_question}")

    standalone_question = {
        "standalone_question": {
            "question": lambda x: rewritten_question, 
            "chat_history": lambda x: x["chat_history"],
        } | CONDENSE_QUESTION_PROMPT | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = final_inputs | ANSWER_PROMPT | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    logging.info("Streaming chain initialized successfully")
    return final_chain.stream({"question": rewritten_question, "memory": memory})

def get_chat_chain(llm, db):
    logging.info("Initializing chat chain")
    retriever = db.as_retriever(search_kwargs={"k": DEFAULT_NUM_RETRIEVED_DOCS})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    def get_rewritten_question(question, chat_history):
        return query_rewriter.rewrite(question)

    standalone_question = {
        "standalone_question": {
            "question": get_rewritten_question,
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        } | CONDENSE_QUESTION_PROMPT | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        logging.info(f"Received question: {question}")
        print(f"Received question: {question}")
        rewritten_question = query_rewriter.rewrite(question)
        logging.info(f"Rewritten Question: {rewritten_question}")
        inputs = {"question": rewritten_question}
        result = final_chain.invoke(inputs)
        logging.info(f"Generated answer: {result['answer']}")
        memory.save_context(inputs, {"answer": result["answer"]})

    logging.info("Chat chain initialized successfully")
    return chat

def get_evaluation_chain(llm, db):
    logging.info("Initializing evaluation chain")
    retriever = db.as_retriever(search_kwargs={"k": DEFAULT_NUM_RETRIEVED_DOCS})

    def evaluate(question: str):
        logging.info(f"Received question: {question}")
        rewritten_question = query_rewriter.rewrite(question)
        logging.info(f"Rewritten Question: {rewritten_question}")
        
        standalone_question = CONDENSE_QUESTION_PROMPT.format(question=rewritten_question, chat_history="")
        standalone_question_result = llm.invoke(standalone_question)
        docs = retriever.invoke({"query": standalone_question_result})
        context = _combine_documents(docs)

        context = _combine_documents(docs)
        
       
        answer_prompt = EVALUATION_PROMPT.format(context=context, question=rewritten_question)
        answer = llm.invoke(answer_prompt)
        logging.info(f"Generated answer: {answer}")
        
        return answer

    logging.info("Evaluation chain initialized successfully")
    return evaluate


def evaluate_predictions(evaluate_fn, test_data: List[Dict]) -> None:
    predictions = []
    for item in test_data:
        question = item["qa"]["question"]
        answer = item["qa"]["answer"]  # 将 reference 改为 answer
        logging.info(f"Processing question: {question}")
        response = evaluate_fn(question)
        predicted_answer = response  # Modify based on how the answer is returned
        predictions.append({
            "id": item["id"],
            "question": question,
            "answer": answer, 
            "predicted": predicted_answer  
        })
    
    # Extract predictions and answers
    answers = [item['answer'] for item in predictions]  
    predictions_text = [item['predicted'] for item in predictions]  
    
   
