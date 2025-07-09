# workflows/daily_flow.py
import random
import logging
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from core import config as config_module
from data_ingestion import arxiv_fetcher
from hrag import metadata_db, vector_db
from workflows.ingestion_flow import process_papers_list
from agents import report_agent, ingestion_agent
from utils import pdf_generator
import arxiv

logger = logging.getLogger(__name__)


def _get_user_preferences() -> List[Dict[str, str]]:
    """从JSON文件加载用户选择的偏好类别。"""
    logger.info(f"正在检查用户偏好文件: {config_module.USER_PREFERENCES_PATH}")
    
    if not config_module.USER_PREFERENCES_PATH.exists():
        logger.warning("诊断：偏好文件 user_preferences.json 不存在。")
        return []
        
    try:
        with open(config_module.USER_PREFERENCES_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                logger.warning("诊断：偏好文件存在但内容为空。")
                return []
            
            logger.info("诊断：偏好文件存在且非空，尝试解析JSON...")
            prefs = json.loads(content)
        
        selected = prefs.get("selected_categories", [])
        logger.info(f"诊断：成功解析JSON，找到 {len(selected)} 条偏好。")
        return selected

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"诊断：读取或解析偏好文件时出错: {e}", exc_info=True)
        return []


def _run_cold_start_population():
    """
    冷启动函数。通过循环N次，每次都从过去几年内随机选择一个年月，
    并从该月中随机获取一篇高质量论文，以建立一个多样化且现代的分类体系。
    """
    current_config = config_module.get_current_config()
    logger.info("--- [冷启动机制触发] ---")
    logger.info("数据库中论文过少，开始逐篇随机采样以建立分类体系。")

    papers_to_process = []
    seen_ids = set()
    target_count = current_config['COLD_START_PAPER_COUNT']
    years_window = current_config['COLD_START_YEARS_WINDOW']
    domains_query = " OR ".join([f"cat:{domain}" for domain in current_config['COLD_START_DOMAINS']])
    
    now = datetime.now()
    all_possible_months = []
    for year in range(now.year - years_window + 1, now.year + 1):
        end_month = now.month if year == now.year else 12
        for month in range(1, end_month + 1):
            all_possible_months.append((year, month))
            
    if not all_possible_months:
        logger.warning("冷启动：未能生成任何可供采样的年月范围。")
        return

    attempts = 0
    max_attempts = target_count * 10 

    while len(papers_to_process) < target_count and attempts < max_attempts:
        attempts += 1
        
        year, month = random.choice(all_possible_months)
        
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        full_query = f"({domains_query}) AND {date_query}"
        
        candidate_papers = arxiv_fetcher.search_arxiv(
            query=full_query,
            max_results=20, 
            sort_by=arxiv.SortCriterion.Relevance
        )

        if not candidate_papers:
            logger.info(f"在 {year}-{month} 未找到任何论文，尝试下一个随机年月。")
            continue

        random.shuffle(candidate_papers)
        for paper in candidate_papers:
            arxiv_id = paper['arxiv_id']
            if arxiv_id not in seen_ids and not metadata_db.check_if_paper_exists(arxiv_id):
                logger.info(f"冷启动：成功采样到新论文 {arxiv_id} (来自 {year}-{month})。 "
                            f"当前进度: {len(papers_to_process) + 1}/{target_count}")
                papers_to_process.append(paper)
                seen_ids.add(arxiv_id)
                break
    
    if not papers_to_process:
        logger.warning("冷启动：在指定次数的尝试中未能获取到任何新的、符合条件的论文。")
        return
        
    logger.info(f"冷启动采样完成，将处理 {len(papers_to_process)} 篇新论文。")

    process_papers_list(papers_to_process, pdf_parsing_strategy=current_config['PDF_PARSING_STRATEGY'])
    logger.info("--- [冷启动机制完成] ---")



def run_daily_workflow():
    """
    执行完整的每日工作流：
    1. (可选) 执行冷启动以填充分类。
    2. 获取新论文。
    3. 对新论文进行独立分类，然后进行分类对齐。
    4. 根据用户的精确分类偏好进行筛选。
    5. 处理并入库筛选后的论文。
    6. 生成报告。
    """
    logger.info("🚀 --- [V2.1 每日工作流启动 - 含分类对齐] --- 🚀")
    
    current_config = config_module.get_current_config()
    logger.info(f"当前任务使用的每日处理上限为: {current_config['DAILY_PAPER_PROCESS_LIMIT']}")

    if metadata_db.get_total_paper_count() < current_config["COLD_START_PAPER_COUNT"]:
        _run_cold_start_population()

    user_preferences = _get_user_preferences()
    if not user_preferences:
        logger.warning("用户未选择任何偏好类别，将不会处理任何论文。请在UI中设置偏好。")
        return {"message": "User preferences not set.", "papers_processed": 0}
    
    pref_set = set((item['domain'], item['task']) for item in user_preferences)
    known_categories = ingestion_agent.get_known_categories()
    logger.info(f"加载了 {len(pref_set)} 条用户偏好和 {len(known_categories)} 个已知领域。")

    try:
        new_papers_data = arxiv_fetcher.fetch_daily_papers(
            domains=current_config["DEFAULT_ARXIV_DOMAINS"]
        )
        if not new_papers_data:
            logger.info("✅ 未发现新论文，每日任务结束。")
            return {"message": "No new papers today.", "papers_processed": 0}
        logger.info(f"发现 {len(new_papers_data)} 篇新论文，开始进行分类、对齐和筛选...")
    except Exception as e:
        logger.critical(f"❌ 获取每日论文时发生严重错误: {e}", exc_info=True)
        return {"message": "Failed to fetch papers from arXiv.", "papers_processed": 0}

    papers_to_process = []
    for paper in new_papers_data:
        arxiv_id = paper['arxiv_id']
        if metadata_db.check_if_paper_exists(arxiv_id):
            continue

        raw_classification = ingestion_agent.classify_paper(paper['title'], paper['summary'])
        if not raw_classification:
            logger.warning(f"无法对论文 {arxiv_id} 进行初步分类，已跳过。")
            continue
        
        aligned_result = ingestion_agent.align_classification(raw_classification, known_categories)
        if not aligned_result:
            logger.warning(f"对齐论文 {arxiv_id} 的分类失败，已跳过。")
            continue

        # vvv [修改] 使用正确的键名并更新分类体系 vvv
        final_domain = aligned_result["final_domain"]
        final_task = aligned_result["final_task"]
        
        # 将对齐后的、标准化的分类更新到全局分类文件中
        ingestion_agent._update_known_categories(final_domain, final_task)

        final_classification = {
            "domain": final_domain,
            "task": final_task
        }
        # ^^^ [修改] ^^^
        
        paper_category = (final_classification['domain'], final_classification['task'])
        if paper_category in pref_set:
            logger.info(f"👍 论文 {arxiv_id} 最终分类为 {paper_category}，匹配用户偏好，加入处理队列。")
            paper['classification_result'] = final_classification
            papers_to_process.append(paper)
        else:
            logger.info(f"👎 论文 {arxiv_id} 最终分类为 {paper_category}，不匹配用户偏好，已跳过。")
    
    if not papers_to_process:
        logger.info("✅ 筛选完成，没有论文匹配用户的偏好类别。每日任务结束。")
        return {"message": "No papers matched user preference.", "papers_processed": 0}
    
    logger.info(f"筛选完成，共找到 {len(papers_to_process)} 篇匹配用户偏好的论文。")
    
    limit = current_config["DAILY_PAPER_PROCESS_LIMIT"]
    if len(papers_to_process) > limit:
        logger.warning(
            f"匹配的论文数 ({len(papers_to_process)}) 超过了设定的上限 "
            f"({limit})，将只处理前 {limit} 篇。"
        )
        papers_to_process = papers_to_process[:limit]

    logger.info(f"最终将有 {len(papers_to_process)} 篇论文进入处理流程。")

    successfully_processed_papers = process_papers_list(
        papers_to_process, 
        pdf_parsing_strategy=current_config["PDF_PARSING_STRATEGY"]
    )
    
    if vector_db.vector_db_manager and successfully_processed_papers:
        logger.info("所有论文处理完毕，正在保存向量索引...")
        vector_db.vector_db_manager.save()
    
    _generate_daily_report(successfully_processed_papers)
    
    final_message = f"每日工作流完成。成功处理并入库 {len(successfully_processed_papers)} 篇论文。"
    logger.info(f"🏁 --- [每日工作流结束]: {final_message} --- 🏁")
    
    return {
        "message": final_message,
        "papers_processed": len(successfully_processed_papers)
    }


def _prepare_report_context(structured_chunks: List[Dict[str, Any]], hrag_manager: 'HRAGManager', arxiv_id: str) -> Dict[str, Any]:
    """
    Prepares a concise and informative context for the report generation agent.
    This includes the paper's summary and a list of images/tables with their surrounding context.
    """
    # 1. 获取已生成的高质量摘要
    paper_summary = hrag_manager.metadata_db.get_paper_summary_by_id(arxiv_id)
    if not paper_summary:
        logger.warning(f"Could not retrieve summary for {arxiv_id}, report quality may be affected.")
        paper_summary = "Summary not available."

    # 2. 提取图片和表格，并捕获其上下文
    media_with_context = []
    text_chunks = [chunk['text'] for chunk in structured_chunks if chunk['type'] == 'text']
    
    for i, chunk in enumerate(structured_chunks):
        chunk_type = chunk.get('type')
        if chunk_type in ['image', 'table']:
            # 提取标题
            caption = chunk.get('metadata', {}).get('caption', 'No caption available.')
            # 提取文件路径 (针对图片)
            image_path = chunk.get('metadata', {}).get('image_path', '')

            # 寻找上下文：向前和向后查找最近的文本块
            context_before = ""
            context_after = ""
            
            # 向前查找
            for j in range(i - 1, -1, -1):
                if structured_chunks[j].get('type') == 'text':
                    context_before = structured_chunks[j].get('text', '')
                    break
            
            # 向后查找
            for j in range(i + 1, len(structured_chunks)):
                if structured_chunks[j].get('type') == 'text':
                    context_after = structured_chunks[j].get('text', '')
                    break
            
            formatted_item = (
                f"- Type: {chunk_type.capitalize()}\n"
                f"  Caption: {caption}\n"
            )
            if image_path:
                formatted_item += f"  Image Path: {image_path}\n"
            
            formatted_item += (
                f"  Context Before: \"...{context_before[-200:]}\"\n"  # 最多截取前文的后200个字符
                f"  Context After: \"{context_after[:200]}...\""      # 最多截取后文的前200个字符
            )
            media_with_context.append(formatted_item)

    media_summary_str = "\n\n".join(media_with_context) if media_with_context else "No images or tables found in the document."

    return {
        "paper_summary": paper_summary,
        "media_summary_str": media_summary_str
    }

def _generate_daily_report(hrag_manager: 'HRAGManager', successful_papers: List[Dict[str, Any]]):
    """Generates and saves a daily report for all newly processed papers."""
    if not successful_papers:
        logger.info("No new papers to generate a report for.")
        return

    logger.info(f"Preparing to generate daily report for {len(successful_papers)} new papers...")
    
    report_agent = ReportAgent()
    report_data_list = []

    for paper_data in successful_papers:
        arxiv_id = paper_data['arxiv_id']
        structured_data_path = paper_data['structured_data_path']

        try:
            with open(structured_data_path, 'r') as f:
                structured_chunks = json.load(f)

            # --- 这是核心修改 ---
            # 准备一个精简、高效的上下文，而不是传递整个（可能被截断的）文件内容
            report_context = _prepare_report_context(structured_chunks, hrag_manager, arxiv_id)
            
            # 将准备好的上下文传递给 report_agent
            report_json = report_agent.generate_report_json_for_paper(
                paper_meta=paper_data, 
                report_context=report_context
            )
            # --------------------

            if report_json:
                report_data_list.append(report_json)

        except FileNotFoundError:
            logger.error(f"Structured data file not found for {arxiv_id} at {structured_data_path}. Skipping for report generation.")
        except Exception as e:
            logger.error(f"Failed to generate report section for {arxiv_id}: {e}", exc_info=True)

    if not report_data_list:
        logger.warning("No report data could be generated for any of the papers.")
        return

    # 保存 JSON 和 PDF 报告的逻辑保持不变
    # ... (这部分逻辑和原来一样) ...
    storage_path = current_config.get('storage_path', 'storage')
    report_dir = os.path.join(storage_path, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    json_report_path = os.path.join(report_dir, f"Daily_arXiv_Report_{today_str}.json")

    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data_list, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON report saved to: {json_report_path}")

    try:
        pdf_generator = PDFGenerator()
        pdf_generator.generate_from_json(json_report_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred during PDF generation: {e}", exc_info=True)