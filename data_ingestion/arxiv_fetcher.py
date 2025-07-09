# data_ingestion/arxiv_fetcher.py

import logging
import arxiv
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# ▼▼▼ [修改] 改变导入方式 ▼▼▼
from core import config as config_module

logger = logging.getLogger(__name__)

def parse_arxiv_result(result: arxiv.Result) -> Dict[str, Any]:
    """将一个 arxiv.Result 对象解析为我们项目内部使用的标准字典格式。"""
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "summary": result.summary,
        "authors": [author.name for author in result.authors],
        "published_date": result.published,
        "updated_date": result.updated,
        "pdf_url": result.pdf_url,
        "primary_category": result.primary_category
    }



def fetch_daily_papers(
    domains: List[str],
    max_results: int = 100
) -> List[Dict[str, Any]]:
    """Fetches the latest papers from the last 3 days in the specified domains."""
    if not domains:
        logger.warning("未提供任何ArXiv领域，无法获取每日论文。")
        return []
        
    query = " OR ".join([f"cat:{domain}" for domain in domains])
    # 保持按提交日期排序，以获取最新的条目
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    start_date_utc = datetime.now(timezone.utc) - timedelta(days=3)
    logger.info(f"Fetching up to {max_results} latest papers published after {start_date_utc.strftime('%Y-%m-%d %H:%M:%S')} (UTC) for domains '{', '.join(domains)}'...")
    
    results = []
    counter = 0
    try:
        for result in search.results():
            counter += 1
            
            # ▼▼▼ 核心修改：使用 .published 替代 .updated ▼▼▼
            # 这确保我们只筛选真正在指定时间窗口内首次发表的论文
            if result.published >= start_date_utc:
                results.append(parse_arxiv_result(result))
            # ▲▲▲ 核心修改结束 ▲▲▲

    except arxiv.UnexpectedEmptyPageError:
        logger.info("Reached the end of the result stream (encountered an empty page), which is normal.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching daily papers: {e}", exc_info=True)
        return results

    logger.info(f"Daily paper fetch complete. Scanned {counter} papers, found {len(results)} recent papers matching the date criteria.")
    return results


def search_arxiv(
    query: str,
    # ▼▼▼ [修改] 修正默认参数问题 ▼▼▼
    max_results: Optional[int] = None,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> List[Dict[str, Any]]:
    """
    Performs an online search based on a constructed query string and specified sorting method.
    """
    if not query:
        logger.warning("No query string provided, cannot perform online search.")
        return []

    # ▼▼▼ [修改] 在函数内部获取配置 ▼▼▼
    if max_results is None:
        max_results = config_module.get_current_config()['USER_QUERY_FETCH_LIMIT']

    logger.info(f"Performing online arXiv search: Query='{query}', SortBy='{sort_by.value}', MaxResults='{max_results}'")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by
    )

    results = []
    try:
        for result in search.results():
            results.append(parse_arxiv_result(result))
    except arxiv.UnexpectedEmptyPageError:
        logger.info("Reached the end of the search result stream (encountered an empty page), which is normal.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during online search: {e}", exc_info=True)
        return results

    logger.info(f"Online search successful, found {len(results)} papers.")
    return results

    
def fetch_papers_by_date_range(
    domains: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 100
) -> List[Dict[str, Any]]:
    """根据指定的领域和时间范围获取论文。"""
    if not domains: return []
    query = " AND ".join([
        f"({' OR '.join([f'cat:{domain}' for domain in domains])})",
        f"submittedDate:[{start_date.strftime('%Y%m%d%H%M%S')} TO {end_date.strftime('%Y%m%d%H%M%S')}]"
    ])
    logger.info(f"正在根据自定义条件搜索论文：领域={domains}, 时间范围=[{start_date.isoformat()}, {end_date.isoformat()}]")
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = [parse_arxiv_result(result) for result in search.results()]
        logger.info(f"自定义搜索成功，获取到 {len(results)} 篇论文。")
        return results
    except Exception as e:
        logger.error(f"自定义搜索时发生错误: {e}", exc_info=True)
        return []
