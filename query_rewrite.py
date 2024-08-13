from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from reranker import Reranker
import logging
from difflib import SequenceMatcher

# 定义重写问句的Prompt
REWRITER_PROMPT = ChatPromptTemplate.from_template(
    """Given a question and its context and a rewrite that decontextualizes the question, 
    edit the rewrite to create a revised version that fully addresses coreferences and omissions 
    in the question without changing the original meaning of the question but providing more information. 
    The new rewrite should not duplicate any previously asked questions in the context. 
    If there is no need to edit the rewrite, return the rewrite as-is.
    Previous Chat History:
    "{chat_history}"
    Current Query:
    "{current}"
    Only output the rewritten question! DON'T add any prefix. Just answer the final result."""
)

class QueryRewriter:
    def __init__(
        self, 
        model_name: str, 
        conversation_id: str,
        memory: ConversationBufferMemory,
        reranker: Reranker,
    ):
        self._llm = Ollama(model=model_name)
        self._conversation_id = conversation_id
        self._memory = memory
        self._output_parser = StrOutputParser()
        self._reranker = reranker

    def rewrite(self, query: str) -> str:
        chain = REWRITER_PROMPT | self._llm | self._output_parser
        
        # 获取最新的对话历史
        memory_variables = self._memory.load_memory_variables({})
        chat_history = memory_variables.get("history", "Empty")
        
        # 调用链条生成重写的查询
        rewrite = chain.invoke({"chat_history": chat_history, "current": query})

        # 打印生成的重写查询结果到日志
        logging.info(f"Original Query: {query}")
        logging.info(f"Rewritten Query: {rewrite}")

        # 评估重写问句的质量
        if not self.is_relevant_rewrite(query, rewrite):
            logging.warning("Rewritten query is not relevant, using the original query.")
            return query
        
        return rewrite

    def is_relevant_rewrite(self, original_query: str, rewritten_query: str) -> bool:
        # 使用字符串相似度来评估重写问句的相关性，确保语义相似
        similarity = self.calculate_similarity(original_query, rewritten_query)
        logging.info(f"Similarity between original and rewritten query: {similarity}")
        return similarity > 0.7  # 可以根据需要调整这个阈值
    
    # 质量检查与相似度计算:

    #使用 SequenceMatcher 来计算原始问句和重写问句之间的相似度，确保重写后的问句与原始问句在语义上保持一致。
    #如果相似度低于一定阈值（如 0.7），系统将保留使用原始问句
    @staticmethod
    def calculate_similarity(str1: str, str2: str) -> float:
        # 使用 SequenceMatcher 来计算两个字符串的相似度
        return SequenceMatcher(None, str1, str2).ratio()
