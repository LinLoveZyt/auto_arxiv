# workflows/daily_flow.py
import random
import logging
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

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


def _generate_daily_report(processed_papers: List[Dict[str, Any]]):
    """为处理过的论文生成包含统计信息的每日报告。"""
    if not processed_papers:
        logger.info("没有新处理的论文，不生成报告。")
        return

    logger.info(f"准备为 {len(processed_papers)} 篇新论文生成每日报告...")
    
    statistics = {}
    total_papers = len(processed_papers)
    for paper in processed_papers:
        classification = paper.get("classification_result", {})
        domain = classification.get("domain", "Unclassified")
        task = classification.get("task", "Unclassified")
        
        if domain not in statistics: statistics[domain] = {}
        if task not in statistics[domain]: statistics[domain][task] = 0
        statistics[domain][task] += 1

    report_jsons = []
    # ▼▼▼ [核心修改] 循环时获取完整的、包含AI摘要的论文数据 ▼▼▼
    for paper_info in processed_papers:
        arxiv_id = paper_info["arxiv_id"]
        
        # 从数据库获取包含AI生成摘要的完整论文详情
        paper_details = metadata_db.get_paper_details_by_id(arxiv_id)
        if not paper_details:
            logger.warning(f"无法从数据库获取论文 {arxiv_id} 的详细信息，跳过此论文的报告生成。")
            continue
            
        # paper_details 现在包含了 'generated_summary' 字段
        
        safe_arxiv_id = arxiv_id.replace('/', '_')
        structured_content_path = config_module.STRUCTURED_DATA_DIR / f"{safe_arxiv_id}.json"
        if not structured_content_path.exists(): 
            logger.warning(f"找不到 {arxiv_id} 的结构化内容文件，跳过此论文的报告生成。")
            continue
            
        with open(structured_content_path, 'r', encoding='utf-8') as f:
            structured_chunks = json.load(f)
            
        report_json_part = report_agent.generate_report_json_for_paper(
            paper_meta=paper_details, # 传递完整的、更新后的论文元数据
            structured_chunks=structured_chunks
        )
        if report_json_part:
            report_jsons.append(report_json_part)
    # ▲▲▲ 修改结束 ▲▲▲

    if not report_jsons:
        logger.error("所有论文的报告内容都生成失败，无法创建每日报告。")
        return
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    final_report_data = {
        "report_title": f"arXiv Daily Focus Report",
        "report_date": today_str,
        "statistics": {"total_papers": total_papers, "breakdown": statistics},
        "papers": report_jsons
    }
    
    report_filename_base = f"{config_module.DAILY_REPORT_PREFIX}_{today_str}"
    json_report_path = config_module.REPORTS_DIR / f"{report_filename_base}.json"
    pdf_report_path = config_module.REPORTS_DIR / f"{report_filename_base}.pdf"

    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report_data, f, ensure_ascii=False, indent=4)
    logger.info(f"JSON报告已保存: {json_report_path}")
    
    # 动态判断报告语言
    report_language = 'zh' if "qwen3" in config_module.get_current_config()['OLLAMA_MODEL_NAME'].lower() else 'en'
    logger.info(f"Generating PDF report in '{report_language}' language.")
    pdf_generator.generate_daily_report_pdf(final_report_data, pdf_report_path, language=report_language)

def run_category_collection_workflow() -> Dict[str, Any]:
    """
    执行一个轻量级的类别收集工作流。
    该流程仅从arXiv获取论文，使用LLM进行分类以丰富本地的`categories.json`文件，
    但不会对论文进行下载、解析或入库，以实现快速的分类体系扩充。
    """
    current_config = config_module.get_current_config()
    logger.info("--- [手动类别收集工作流启动] ---")

    target_count = current_config['CATEGORY_COLLECTION_COUNT']
    years_window = current_config['CATEGORY_COLLECTION_YEARS_WINDOW']
    domains_query = " OR ".join([f"cat:{domain}" for domain in current_config['CATEGORY_COLLECTION_DOMAINS']])
    
    logger.info(f"目标：收集 {target_count} 篇论文的分类信息。")

    now = datetime.now()
    all_possible_months = []
    for year in range(now.year - years_window + 1, now.year + 1):
        end_month = now.month if year == now.year else 12
        for month in range(1, end_month + 1):
            all_possible_months.append((year, month))
            
    if not all_possible_months:
        logger.warning("类别收集：未能生成任何可供采样的年月范围。")
        return {"message": "未能生成任何可供采样的年月范围。", "categories_added": 0}

    papers_to_classify = []
    seen_ids = set()
    attempts = 0
    max_attempts = target_count * 15 # 增加尝试次数以提高成功率

    while len(papers_to_classify) < target_count and attempts < max_attempts:
        attempts += 1
        year, month = random.choice(all_possible_months)
        start_date = datetime(year, month, 1)
        end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        full_query = f"({domains_query}) AND {date_query}"
        
        # 每次只获取少量候选，避免单次API调用过多
        candidate_papers = arxiv_fetcher.search_arxiv(
            query=full_query, max_results=20, sort_by=arxiv.SortCriterion.Relevance
        )

        if not candidate_papers: continue

        random.shuffle(candidate_papers)
        for paper in candidate_papers:
            arxiv_id = paper['arxiv_id']
            # 我们只关心我们还不知道分类的论文，但这里简化为检查论文是否存在
            if arxiv_id not in seen_ids and not metadata_db.check_if_paper_exists(arxiv_id):
                logger.info(f"类别收集：成功采样到新论文 {arxiv_id} (来自 {year}-{month})。 "
                            f"当前进度: {len(papers_to_classify) + 1}/{target_count}")
                papers_to_classify.append(paper)
                seen_ids.add(arxiv_id)
                if len(papers_to_classify) >= target_count:
                    break
        if len(papers_to_classify) >= target_count:
            break
    
    if not papers_to_classify:
        logger.warning("类别收集：在指定次数的尝试中未能获取到任何新的、符合条件的论文。")
        return {"message": "未能获取到任何新的论文。", "categories_added": 0}
        
    logger.info(f"采样完成，将对 {len(papers_to_classify)} 篇新论文进行纯分类处理...")

    new_categories_count = 0
    known_categories = ingestion_agent.get_known_categories()

    for paper in papers_to_classify:
        # 这是核心的简化：只进行分类，不入库
        raw_classification = ingestion_agent.classify_paper(paper['title'], paper['summary'])
        if not raw_classification:
            logger.warning(f"无法对论文 {paper['arxiv_id']} 进行分类，已跳过。")
            continue
        
        # 可选：进行分类对齐以保持体系整洁
        aligned_result = ingestion_agent.align_classification(raw_classification, known_categories)
        if aligned_result:
             # _update_known_categories 已在 classify_paper 和 align_classification 内部被调用
             new_categories_count += 1
             # 更新内存中的 known_categories 以供下一次对齐使用
             ingestion_agent._update_known_categories(aligned_result["final_domain"], aligned_result["final_task"])


    message = f"类别收集完成！成功处理了 {len(papers_to_classify)} 篇论文，扩充了分类体系。"
    logger.info(f"--- [手动类别收集工作流结束]: {message} ---")
    return {"message": message, "categories_added": new_categories_count}
