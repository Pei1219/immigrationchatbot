from typing import Union, List
from dotenv import load_dotenv
import os
from default_config import *
from langchain_community.llms import Ollama


class Singleton(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance[cls]
    
class BaseSingleton(metaclass=Singleton):
    pass

class BaseObject(BaseSingleton):
    @classmethod
    def class_name(cls):
        return cls.__name__
  

    

class Config(BaseObject):
    def __init__(
        self,
        # Ollama LLM Settings
        llm_name: str = "llama2:7b-chat",
        enable_history: bool = True,
        # Reranker Settings
        rerank_model_name: str = "BAAI/bge-reranker-large",
        similarity_top_k: int = 10,
        rerank_top_k: int = 3,
     
    ):
        super().__init__()
        load_dotenv('.env')

        # LLM Settings
        self.llm_name = llm_name if llm_name is not None else os.getenv(LLM_NAME)
        self._llm = Ollama(model=llm_name) 
        self.enable_history = enable_history if enable_history is not None else os.getenv(ENABLE_HISTROY)

        # Reranker Settings
        self.rerank_model_name = rerank_model_name if rerank_model_name is not None else os.getenv(RERANK_MODEL_NAME)
        self.similarity_top_k = similarity_top_k if similarity_top_k is not None else os.getenv(SIMILARITY_TOP_K)
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else os.getenv(RERANK_TOP_K)
