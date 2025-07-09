# workflows/ingestion_flow.py

import logging
import json
from typing import List, Dict, Any

from hrag import metadata_db
from data_ingestion import pdf_processor
from hrag import hrag_manager as hrag_manager_module
import time
import gc
import torch

logger = logging.getLogger(__name__)

def process_papers_list(
    papers_to_process: List[Dict[str, Any]],
    pdf_parsing_strategy: str = "monkey"  # 默认使用 'monkey'
) -> List[Dict[str, Any]]:
    """
    处理一个给定的论文信息字典列表，将它们下载、解析并添加到知识库中。
    这是一个可复用的核心工作流。

    Args:
        papers_to_process: 一个包含多篇论文信息的列表。
        pdf_parsing_strategy: 使用的PDF解析策略 ('monkey' 或 'fast')。

    Returns:
        一个列表，包含被成功处理并入库的论文信息。
    """
    if not papers_to_process:
        return []

    logger.info(f"--- [通用入库流程启动]：准备处理 {len(papers_to_process)} 篇论文，使用 '{pdf_parsing_strategy}' 解析策略 ---")
    
    successfully_processed_papers = []
    
    for i, paper_data in enumerate(papers_to_process, 1):
        arxiv_id = paper_data["arxiv_id"]
        title = paper_data["title"]
        
        logger.info(f"--- [进度 {i}/{len(papers_to_process)}] 开始处理论文: {arxiv_id} - {title[:60]}... ---")

        try:
            if metadata_db.check_if_paper_exists(arxiv_id):
                logger.info(f"论文 {arxiv_id} 已存在于数据库中，跳过。")
                successfully_processed_papers.append(paper_data)
                continue

            # ▼▼▼ [修改] 将解析策略作为参数传递 ▼▼▼
            path_info = pdf_processor.process_paper(paper_data, strategy=pdf_parsing_strategy)
            if not path_info:
                logger.error(f"处理论文 {arxiv_id} 的PDF失败，跳过此论文。")
                continue
            
            # ... 函数的其余部分保持不变 ...
            full_paper_data = {**paper_data, **path_info}
            paper_db_id = metadata_db.add_paper(full_paper_data)
            if paper_db_id is None:
                logger.error(f"无法将论文 {arxiv_id} 的元数据初始写入数据库，跳过。")
                continue

            with open(path_info["json_path"], 'r', encoding='utf-8') as f:
                structured_chunks = json.load(f)

            success = hrag_manager_module.hrag_manager.process_and_add_paper(
                paper_data, 
                structured_chunks,
                classification=paper_data.get('classification_result')
            )
            if success:
                successfully_processed_papers.append(paper_data)
                logger.info(f"🎉 论文 {arxiv_id} 已成功完成所有处理并入库！")
            else:
                logger.error(f"H-RAG管理器未能成功处理论文 {arxiv_id}，请检查相关日志。")

        except Exception as e:
            logger.critical(f"❌ 处理论文 {arxiv_id} 时发生意外的严重错误，跳过此论文: {e}", exc_info=True)
            continue
        
        finally:
            logger.info(f"--- [循环间隙清理] 正在为下一篇论文准备环境... ---")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)

    logger.info(f"--- [通用入库流程结束]：成功处理 {len(successfully_processed_papers)} / {len(papers_to_process)} 篇论文 ---")
    return successfully_processed_papers