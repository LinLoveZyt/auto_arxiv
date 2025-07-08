# core/bootstrap.py

import logging
from core.logger import setup_logging
from core.llm_client import initialize_llm_client
from hrag.embedding_engine import initialize_embedding_engine
from hrag.vector_db import initialize_vector_db
from hrag.hrag_manager import initialize_hrag_manager
from hrag.reranker import initialize_reranker

logger = logging.getLogger(__name__)

def initialize_core_services():
    """
    按正确依赖顺序，统一初始化所有核心服务。
    这个函数是幂等的，可以安全地多次调用。
    """
    logger.info("--- [引导程序]：开始初始化所有核心服务... ---")
    
    # 日志系统是第一个，尽管它通常在主入口就调用了
    # setup_logging() # 通常在 cli() 回调中已完成，此处调用是可选的，用于确保
    
    # 核心依赖顺序：嵌入引擎 -> 向量数据库 -> LLM客户端 -> Reranker -> H-RAG管理器
    initialize_embedding_engine()
    initialize_vector_db()
    initialize_llm_client()
    initialize_reranker()
    initialize_hrag_manager()
    
    logger.info("--- [引导程序]：✅ 所有核心服务已成功初始化。 ---")