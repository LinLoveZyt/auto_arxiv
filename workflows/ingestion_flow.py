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
    pdf_parsing_strategy: str = "monkey",
    ingestion_mode: str = 'full'  # 新增参数: 'full' 或 'metadata_only'
) -> List[Dict[str, Any]]:
    """
    处理一个给定的论文信息字典列表。
    这是一个可复用的核心工作流，现在支持两种模式：
    - 'full': 下载、解析PDF，并将所有内容（包括向量）添加到知识库。
    - 'metadata_only': 只将论文的元数据（标题、摘要、分类等）添加到数据库，不处理PDF和向量。
    """
    if not papers_to_process:
        return []

    logger.info(f"--- [通用入库流程启动]：准备处理 {len(papers_to_process)} 篇论文，使用 '{ingestion_mode}' 模式 ---")
    
    successfully_processed_papers = []
    
    for i, paper_data in enumerate(papers_to_process, 1):
        arxiv_id = paper_data["arxiv_id"]
        title = paper_data["title"]
        classification_result = paper_data.get('classification_result') # 获取分类结果

        logger.info(f"--- [进度 {i}/{len(papers_to_process)}] 开始处理论文: {arxiv_id} - {title[:60]}... ---")

        try:
            if metadata_db.check_if_paper_exists(arxiv_id):
                logger.info(f"论文 {arxiv_id} 已存在于数据库中，跳过。")
                # 即使跳过，也认为它是“成功”的，因为它已在库中
                successfully_processed_papers.append(paper_data)
                continue

            # --- 模式分支 ---
            if ingestion_mode == 'full':
                # 完整的PDF处理和入库流程
                path_info = pdf_processor.process_paper(paper_data, strategy=pdf_parsing_strategy)
                if not path_info:
                    logger.error(f"处理论文 {arxiv_id} 的PDF失败，跳过此论文。")
                    continue
                
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
                    classification=classification_result
                )
                if success:
                    successfully_processed_papers.append(paper_data)
                    logger.info(f"🎉 论文 {arxiv_id} 已成功完成所有处理并入库！")
                else:
                    logger.error(f"H-RAG管理器未能成功处理论文 {arxiv_id}，请检查相关日志。")

            elif ingestion_mode == 'metadata_only':
                # 轻量级的元数据入库流程
                logger.info(f"执行轻量级入库，只保存论文 {arxiv_id} 的元数据和分类信息。")
                
                # 1. 添加论文元数据，路径字段为空
                # 我们传入一个空的 path_info，确保 add_paper 函数有东西解包
                path_info = {"pdf_path": None, "json_path": None}
                full_paper_data = {**paper_data, **path_info}
                paper_db_id = metadata_db.add_paper(full_paper_data)
                if paper_db_id is None:
                    logger.error(f"无法将论文 {arxiv_id} 的元数据写入数据库，跳过。")
                    continue
                
                # 2. 如果有分类结果，则更新论文的分类
                if classification_result:
                    conn = metadata_db.get_db_connection()
                    try:
                        with conn:
                            domain_id = metadata_db.add_or_get_domain(classification_result["domain"], conn=conn)
                            task_id = metadata_db.add_or_get_task(classification_result["task"], domain_id, conn=conn)
                            # 只更新分类，摘要字段传None
                            metadata_db.update_paper_summary_and_classification(
                                arxiv_id=arxiv_id, domain_id=domain_id, task_id=task_id, summary=None, conn=conn
                            )
                        logger.info(f"论文 {arxiv_id} 的分类信息已更新。")
                    finally:
                        if conn: conn.close()
                
                successfully_processed_papers.append(paper_data)
                logger.info(f"🎉 论文 {arxiv_id} 的元数据已成功入库！")
            
            else:
                 logger.error(f"未知的入库模式: '{ingestion_mode}'，跳过论文 {arxiv_id}。")
                 continue

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
