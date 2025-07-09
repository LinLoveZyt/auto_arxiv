# hrag/hrag_manager.py
import logging
import json
import torch # 导入torch以使用其cuda功能
from typing import List, Dict, Any, Optional

from core import config as config_module
from hrag import metadata_db, vector_db
from hrag import embedding_engine as embedding_engine_module 
from agents import ingestion_agent, summarization_agent
import gc

logger = logging.getLogger(__name__)

class HRAGManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HRAGManager, cls).__new__(cls)
        return cls._instance
        
    # hrag/hrag_manager.py

    def process_and_add_paper(
        self,
        paper_data: Dict[str, Any],
        structured_chunks: List[Dict[str, Any]],
        classification: Optional[Dict[str, Any]] = None
    ) -> bool:
        arxiv_id, title, abstract = paper_data["arxiv_id"], paper_data["title"], paper_data["summary"]
        logger.info(f"--- H-RAG管理器开始处理论文: {arxiv_id} ---")

        # 关键修改：从全局配置读取批次大小
        current_config = config_module.get_current_config()
        embedding_batch_size = current_config.get('EMBEDDING_BATCH_SIZE', 64)

        try:
            if classification is None:
                logger.info(f"未提供预分类结果，为论文 {arxiv_id} 独立执行分类...")
                classification = ingestion_agent.classify_paper(title, abstract)
            
            if not classification: 
                raise ValueError("论文分类失败")
            
            paper_summary = summarization_agent.summarize_paper_from_chunks(structured_chunks, title)
            if not paper_summary: 
                raise ValueError("生成论文摘要失败")
            
            all_text_chunks = [paper_summary] + [c['text'] for c in structured_chunks if c.get('text')]
            if not embedding_engine_module.embedding_engine: 
                raise RuntimeError("Embedding Engine 未初始化！")

        except Exception as e:
            logger.error(f"论文 {arxiv_id} 在预处理阶段失败: {e}", exc_info=True)
            return False

        conn = metadata_db.get_db_connection()
        try:
            with conn:
                domain_id = metadata_db.add_or_get_domain(classification["domain"], conn=conn)
                task_id = metadata_db.add_or_get_task(classification["task"], domain_id, conn=conn)
                
                metadata_db.update_paper_summary_and_classification(
                    arxiv_id=arxiv_id, domain_id=domain_id, task_id=task_id, summary=paper_summary, conn=conn
                )

            logger.info(f"开始为论文 {arxiv_id} 的 {len(all_text_chunks)} 个文本块进行分批嵌入和入库 (批次大小: {embedding_batch_size})...")
            
            for i in range(0, len(all_text_chunks), embedding_batch_size):
                batch_texts = all_text_chunks[i:i+embedding_batch_size]
                num_batches = (len(all_text_chunks) + embedding_batch_size - 1) // embedding_batch_size
                logger.info(f"  处理批次 {i//embedding_batch_size + 1}/{num_batches}，包含 {len(batch_texts)} 个文本块...")
                
                # a. 编码
                vectors = embedding_engine_module.embedding_engine.encode(batch_texts, batch_size=32)
                if vectors.size == 0:
                    logger.warning(f"批次 {i//embedding_batch_size + 1} 编码失败，跳过此批次。")
                    continue
                
                with conn:
                    max_id_in_db = metadata_db.get_max_vector_id(conn=conn)
                    start_id = (max_id_in_db + 1) if max_id_in_db is not None else 0
                    
                    metadata_for_batch = []
                    for j, chunk_text in enumerate(batch_texts):
                        is_summary = (i == 0 and j == 0)
                        chunk_index_in_paper = i + j
                        
                        meta = {
                            "id": start_id + j, "type": "paper_summary" if is_summary else "raw_chunk",
                            "source_id": arxiv_id, "chunk_seq": None if is_summary else (chunk_index_in_paper - 1),
                            "domain_id": domain_id, "task_id": task_id, "content_preview": chunk_text[:200]
                        }
                        metadata_for_batch.append(meta)

                    vector_db.vector_db_manager.add(vectors)
                    metadata_db.add_vector_metadata_batch(metadata_for_batch, conn=conn)
                
                # 关键修改：加强循环内的清理
                del batch_texts, vectors, metadata_for_batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info(f"--- ✅ 论文 {arxiv_id} 已成功完成所有文本块的嵌入和入库 ---")
            return True

        except Exception as e:
            logger.critical(f"❌ 处理论文 {arxiv_id} 的数据库事务或嵌入流程中发生严重错误: {e}", exc_info=True)
            return False
        finally:
            if conn: conn.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"已清理PyTorch CUDA缓存并执行垃圾回收，为下一任务释放显存。")
                
# ... initialize_hrag_manager 函数不变 ...
hrag_manager: Optional[HRAGManager] = None
def initialize_hrag_manager():
    global hrag_manager
    if hrag_manager is None: hrag_manager = HRAGManager()
    return hrag_manager