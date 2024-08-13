import sys
import json
import logging
import os
import pandas as pd
from default_config import parse_arguments, DEFAULT_MODEL, DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_API_BASE
from model_utils import check_model_availability
from document_loader import load_documents_into_database
from llm_chain import get_chat_chain, get_evaluation_chain
from langchain_community.llms import Ollama
from langchain_community.llms.openai import OpenAI
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    args = parse_arguments()

    if args.model != "OpenAI":
        try:
            logging.info(f"Checking availability of model: {args.model}")
            check_model_availability(args.model)
            logging.info(f"Checking availability of embedding model: {args.embedding_model}")
            check_model_availability(args.embedding_model)
        except Exception as e:
            logging.error(e)
            sys.exit()

    try:
        logging.info(f"Loading documents from path: {args.path}")
        db = load_documents_into_database(args.embedding_model, args.path, args.chunk_size, args.chunk_overlap)
        logging.info("Documents loaded and database created successfully")
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit()

    if args.model == "OpenAI":
        llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=args.openai_key, openai_api_base=args.openai_base)
    else:
        llm = Ollama(model=args.model)

    if args.eval:
        evaluate_chain = get_evaluation_chain(llm, db)
    
        current_dir = os.path.dirname(__file__)
        json_file_path = os.path.join(current_dir, "evaluation", "FAQ.json")

        with open(json_file_path, "r") as f:
            test_data = json.load(f)
        
        contexts = []
        predictions = []

        # Initialize a counter to track the number of questions processed
        counter = 0

        for item in test_data["FAQs"]:
            if counter >= 2:  # Stop after processing two questions
                break
            
            question = item["question"]
            context = item.get("context", "")
            logging.info(f"Processing question: {question}")
            response = evaluate_chain(question)
            predictions.append(response)
            contexts.append(context)
            
            counter += 1  # Increment the counter
        
        # Convert data to DataFrame
        df = pd.DataFrame({
            "contexts": contexts,
            "predictions": predictions
        })

        # Evaluate using RAGAS
        ragas_results = evaluate(
            df,  # Use DataFrame as input
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )
        logging.info(f"RAGAS Results: {ragas_results}")
        
        eval_results = {
            "ragas_results": ragas_results
        }
        
        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)
        
        logging.info("Evaluation results saved to evaluation/evaluation_results.json")

    else:
        chat = get_chat_chain(llm, db)
        while True:
            try:
                user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
                if user_input.lower() == "exit":
                    break
                response = chat(user_input)
                logging.info(f"User question: {user_input}")
                logging.info(f"Assistant response: {response}")

            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
