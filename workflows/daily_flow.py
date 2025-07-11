# workflows/daily_flow.py
import random
import logging
import json
import time
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Any, Optional
import os


from core import config as config_module
from data_ingestion import arxiv_fetcher, pdf_processor
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


def run_daily_workflow(
    research_plan: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """
    执行完整的每日工作流 (V5.3: OCR后置质量筛选)
    """
    logger.info("🚀 --- [V5.3 每日工作流启动 - OCR后置质量筛选] --- 🚀")
    
    current_config = config_module.get_current_config()
    limit = current_config['DAILY_PAPER_PROCESS_LIMIT']
    
    # --- 阶段1: 兴趣初筛 (无上限) ---
    logger.info("--- [阶段 1/2] 开始进行无上限兴趣筛选 ---")
    
    # ... [这部分获取偏好和日期的代码与之前版本相同，保持不变] ...
    if research_plan:
        logger.info(f"收到本次动态调研计划: '{research_plan[:100]}...'")
    user_preferences = _get_user_preferences()
    if not user_preferences and not research_plan:
        logger.warning("用户未设置任何固定偏好，也未提供动态调研计划。工作流终止。")
        return {"message": "User preferences and research plan are not set.", "papers_processed": 0}
    pref_set = set((item['domain'], item['task']) for item in user_preferences)
    start_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc) if start_date else None
    end_datetime = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc) if end_date else None

    papers_passed_interest_filter = []
    try:
        paper_generator = arxiv_fetcher.fetch_daily_papers(
            domains=current_config["DEFAULT_ARXIV_DOMAINS"],
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        for paper in paper_generator:
            arxiv_id = paper['arxiv_id']
            if metadata_db.check_if_paper_exists(arxiv_id):
                continue
            
            final_classification = ingestion_agent.classify_paper_with_rag_context(paper['title'], paper['summary'])
            if not final_classification: continue
            
            paper['classification_result'] = final_classification
            paper_category = (final_classification['domain'], final_classification['task'])
            
            is_interest_match = False
            interest_reason = ""
            if paper_category in pref_set:
                is_interest_match = True
                interest_reason = f"匹配固定偏好: {paper_category[0]}/{paper_category[1]}"
            elif research_plan:
                is_relevant, justif = ingestion_agent.evaluate_relevance_by_research_plan(paper, research_plan)
                if is_relevant:
                    is_interest_match = True
                    interest_reason = f"匹配调研计划: {justif}"

            if is_interest_match:
                paper['selection_reason'] = interest_reason
                papers_passed_interest_filter.append(paper)
                logger.info(f"👍 论文 {arxiv_id} 通过兴趣初筛，进入待定队列。")
        
        logger.info(f"✅ 兴趣初筛完成，共有 {len(papers_passed_interest_filter)} 篇论文进入下一轮质量筛选。")

    except Exception as e:
        logger.critical(f"❌ 阶段1（兴趣筛选）发生严重错误: {e}", exc_info=True)
        return {"message": "Failed during interest filtering stage.", "papers_processed": 0}

    if not papers_passed_interest_filter:
        logger.info("没有论文通过兴趣筛选，每日任务结束。")
        return {"message": "No papers passed interest filter.", "papers_processed": 0}

    # --- 阶段2: OCR后置质量筛选 (有上限) ---
    logger.info(f"--- [阶段 2/2] 开始对 {len(papers_passed_interest_filter)} 篇论文进行逐一OCR和质量筛选 (上限: {limit}篇) ---")
    
    final_papers_for_report = []
    for i, paper_meta in enumerate(papers_passed_interest_filter):
        arxiv_id = paper_meta['arxiv_id']
        logger.info(f"--- [处理进度 {i+1}/{len(papers_passed_interest_filter)}] 开始处理论文: {arxiv_id} ---")

        if len(final_papers_for_report) >= limit:
            logger.info(f"已达到每日处理上限 ({limit})，终止质量筛选流程。")
            break

        # a. 下载和OCR
        path_info = pdf_processor.process_paper(paper_meta, strategy=current_config["PDF_PARSING_STRATEGY"])
        if not path_info or not path_info.get("json_path"):
            logger.error(f"处理论文 {arxiv_id} 的PDF失败，跳过。")
            continue
        
        # b. 从OCR结果提取作者/机构
        try:
            with open(path_info["json_path"], 'r', encoding='utf-8') as f:
                structured_chunks = json.load(f)
            ocr_authors, ocr_affiliations = ingestion_agent.extract_authors_and_affiliations_from_ocr(structured_chunks)
        except Exception as e:
            logger.error(f"解析OCR JSON或提取信息时失败: {e}", exc_info=True)
            ocr_authors, ocr_affiliations = [], []

        # c. 质量检查
        has_strong_team, team_reason = ingestion_agent.check_strong_team(ocr_affiliations)
        if has_strong_team:
            paper_meta['selection_reason'] += f"\n质量评估: {team_reason}"
            final_papers_for_report.append(paper_meta)
            logger.info(f"✅ 论文 {arxiv_id} 通过强团队筛选，加入最终报告列表。")
            continue

        has_strong_author, author_reason = ingestion_agent.check_strong_author(paper_meta, ocr_authors)
        if has_strong_author:
            paper_meta['selection_reason'] += f"\n质量评估: {author_reason}"
            final_papers_for_report.append(paper_meta)
            logger.info(f"✅ 论文 {arxiv_id} 通过强作者筛选，加入最终报告列表。")
            continue

        # d. 决策：未通过则删除文件
        logger.warning(f"❌ 论文 {arxiv_id} 未通过质量筛选，将删除相关文件。")
        try:
            if path_info.get("pdf_path") and os.path.exists(path_info["pdf_path"]):
                os.remove(path_info["pdf_path"])
            if os.path.exists(path_info["json_path"]):
                os.remove(path_info["json_path"])
            logger.info(f"已清理论文 {arxiv_id} 的PDF和JSON文件。")
        except OSError as e:
            logger.error(f"清理文件时出错: {e}")

    # --- 阶段3: 入库与报告 ---
    if not final_papers_for_report:
        logger.info("✅ 质量筛选完成，没有论文通过。每日任务结束。")
        return {"message": "No papers passed quality filter.", "papers_processed": 0}

    logger.info(f"✅ 质量筛选完成，最终有 {len(final_papers_for_report)} 篇论文将入库并生成报告。")

    successfully_processed_papers = process_papers_list(
        final_papers_for_report, 
        pdf_parsing_strategy=current_config["PDF_PARSING_STRATEGY"],
        ingestion_mode='full'
    )
    
    if vector_db.vector_db_manager and successfully_processed_papers:
        vector_db.vector_db_manager.save()
    
    _generate_daily_report(successfully_processed_papers)
    
    ingestion_agent.export_categories_to_json()
    
    final_message = f"每日工作流完成。成功处理并入库 {len(successfully_processed_papers)} 篇论文。"
    logger.info(f"🏁 --- [每日工作流结束]: {final_message} --- 🏁")
    
    return {"message": final_message, "papers_processed": len(successfully_processed_papers)}


def _generate_daily_report(processed_papers: List[Dict[str, Any]]):
    """为处理过的论文生成包含统计信息和目录的每日报告。"""
    if not processed_papers:
        logger.info("没有新处理的论文，不生成报告。")
        return

    logger.info(f"准备为 {len(processed_papers)} 篇新论文生成每日报告...")
    
    report_jsons_grouped = {}
    
    for paper_info in processed_papers:
        arxiv_id = paper_info["arxiv_id"]
        
        paper_details = metadata_db.get_paper_details_by_id(arxiv_id)
        if not paper_details:
            logger.warning(f"无法从数据库获取论文 {arxiv_id} 的详细信息，跳过此论文的报告生成。")
            continue
            
        paper_details['selection_reason'] = paper_info.get('selection_reason')
            
        safe_arxiv_id = arxiv_id.replace('/', '_')
        structured_content_path = config_module.STRUCTURED_DATA_DIR / f"{safe_arxiv_id}.json"
        if not structured_content_path.exists(): 
            logger.warning(f"找不到 {arxiv_id} 的结构化内容文件，跳过此论文的报告生成。")
            continue
            
        with open(structured_content_path, 'r', encoding='utf-8') as f:
            structured_chunks = json.load(f)
            
        report_json_part = report_agent.generate_report_json_for_paper(
            paper_meta=paper_details,
            structured_chunks=structured_chunks
        )
        if report_json_part:
            
            domain = report_json_part.get('classification', {}).get('domain', 'Unclassified')
            task = report_json_part.get('classification', {}).get('task', 'Unclassified')
            if domain not in report_jsons_grouped:
                report_jsons_grouped[domain] = {}
            if task not in report_jsons_grouped[domain]:
                report_jsons_grouped[domain][task] = []
            report_jsons_grouped[domain][task].append(report_json_part)
            

    if not report_jsons_grouped:
        logger.error("所有论文的报告内容都生成失败，无法创建每日报告。")
        return
    
    
    statistics = {}
    total_papers = 0
    for domain, tasks in report_jsons_grouped.items():
        statistics[domain] = {}
        for task, papers in tasks.items():
            count = len(papers)
            statistics[domain][task] = count
            total_papers += count
    

    today_str = datetime.now().strftime("%Y-%m-%d")
    final_report_data = {
        "report_title": f"arXiv Daily Focus Report",
        "report_date": today_str,
        "statistics": {"total_papers": total_papers, "breakdown": statistics},
        "papers_grouped": report_jsons_grouped
    }
    
    report_filename_base = f"{config_module.DAILY_REPORT_PREFIX}_{today_str}"
    json_report_path = config_module.REPORTS_DIR / f"{report_filename_base}.json"
    pdf_report_path = config_module.REPORTS_DIR / f"{report_filename_base}.pdf"

    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report_data, f, ensure_ascii=False, indent=4)
    logger.info(f"JSON报告已保存: {json_report_path}")
    
    report_language = 'zh' if "qwen" in config_module.get_current_config()['OLLAMA_MODEL_NAME'].lower() else 'en'
    logger.info(f"Generating PDF report in '{report_language}' language.")
    pdf_generator.generate_daily_report_pdf(final_report_data, pdf_report_path, language=report_language)


def run_category_collection_workflow() -> Dict[str, Any]:
    """
    执行一个轻量级的类别收集工作流（V2版：轻量级入库）。
    该流程从arXiv获取论文，使用LLM进行高质量分类，然后只将论文元数据和分类
    信息写入数据库，不处理PDF，以实现快速的分类体系扩充。
    """
    current_config = config_module.get_current_config()
    logger.info("--- [手动类别收集工作流启动 (V2: 轻量级入库)] ---")

    target_count = current_config['CATEGORY_COLLECTION_COUNT']
    years_window = current_config['CATEGORY_COLLECTION_YEARS_WINDOW']
    domains_query = " OR ".join([f"cat:{domain}" for domain in current_config['CATEGORY_COLLECTION_DOMAINS']])
    
    logger.info(f"目标：为 {target_count} 篇论文进行分类并轻量级入库。")

    # ... [此部分采样逻辑与旧代码相同，无需更改] ...
    now = datetime.now()
    all_possible_months = []
    for year in range(now.year - years_window + 1, now.year + 1):
        end_month = now.month if year == now.year else 12
        for month in range(1, end_month + 1):
            all_possible_months.append((year, month))
            
    if not all_possible_months:
        logger.warning("类别收集：未能生成任何可供采样的年月范围。")
        return {"message": "未能生成任何可供采样的年月范围。", "categories_added": 0}

    papers_to_classify_and_ingest = []
    seen_ids = set()
    attempts = 0
    max_attempts = target_count * 15

    while len(papers_to_classify_and_ingest) < target_count and attempts < max_attempts:
        attempts += 1
        year, month = random.choice(all_possible_months)
        start_date = datetime(year, month, 1)
        end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        full_query = f"({domains_query}) AND {date_query}"
        
        candidate_papers = arxiv_fetcher.search_arxiv(
            query=full_query, max_results=20, sort_by=arxiv.SortCriterion.Relevance
        )

        if not candidate_papers: continue

        random.shuffle(candidate_papers)
        for paper in candidate_papers:
            arxiv_id = paper['arxiv_id']
            if arxiv_id not in seen_ids and not metadata_db.check_if_paper_exists(arxiv_id):
                logger.info(f"类别收集：成功采样到新论文 {arxiv_id} (来自 {year}-{month})。 "
                            f"当前进度: {len(papers_to_classify_and_ingest) + 1}/{target_count}")
                papers_to_classify_and_ingest.append(paper)
                seen_ids.add(arxiv_id)
                if len(papers_to_classify_and_ingest) >= target_count:
                    break
        if len(papers_to_classify_and_ingest) >= target_count:
            break
    
    if not papers_to_classify_and_ingest:
        logger.warning("类别收集：在指定次数的尝试中未能获取到任何新的、符合条件的论文。")
        return {"message": "未能获取到任何新的论文。", "categories_added": 0}
        
    logger.info(f"采样完成，将对 {len(papers_to_classify_and_ingest)} 篇新论文进行高级分类和轻量级入库...")

    # 对采样的论文进行分类
    for paper in papers_to_classify_and_ingest:
        # 使用新的高级分类函数，保证分类质量
        final_classification = ingestion_agent.classify_paper_with_rag_context(paper['title'], paper['summary'])
        if final_classification:
            paper['classification_result'] = final_classification
        else:
            # 如果分类失败，可以给一个默认值或直接从列表中移除
            logger.warning(f"论文 {paper['arxiv_id']} 分类失败，将不会被入库。")
    
    # 过滤掉分类失败的论文
    papers_to_ingest_lightly = [p for p in papers_to_classify_and_ingest if 'classification_result' in p]
    
    successfully_processed = process_papers_list(
        papers_to_ingest_lightly,
        ingestion_mode='metadata_only'
    )

    # 在工作流最后，同步一次分类体系，确保UI显示最新
    ingestion_agent.export_categories_to_json()

    message = f"类别收集完成！成功为 {len(successfully_processed)} 篇论文添加了分类并轻量级入库。"
    logger.info(f"--- [手动类别收集工作流结束]: {message} ---")
    return {"message": message, "categories_added": len(successfully_processed)}
