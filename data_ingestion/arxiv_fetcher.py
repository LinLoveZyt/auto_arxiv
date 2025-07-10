# data_ingestion/arxiv_fetcher.py

import logging
import arxiv
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Generator

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
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    根据指定的领域和时间范围获取论文，并作为生成器逐一返回。
    使用经过验证的`arxiv.Client`和异常捕获机制，以实现稳健、可靠的无限分页。
    """
    if not domains:
        logger.warning("未提供任何ArXiv领域，无法获取论文。")
        return

    # 1. 设置默认日期和查询条件
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=3)

    query = " OR ".join([f"cat:{domain}" for domain in domains])
    date_query = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M%S')} TO {end_date.strftime('%Y%m%d%H%M%S')}]"
    full_query = f"({query}) AND {date_query}"
    
    logger.info(f"正在根据以下条件搜索论文：Query='{full_query}'")

    # 2. 构造 Client (定义获取方式) 和 Search (定义获取内容)
    client = arxiv.Client(
      page_size = 1000,
      delay_seconds = 5,
      num_retries = 5
    )
    search = arxiv.Search(
        query=full_query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    total_found = 0
    try:
        # 3. 调用 client.results() 获取生成器，并优雅地处理已知异常
        results_generator = client.results(search)
        for result in results_generator:
            yield parse_arxiv_result(result)
            total_found += 1
            if total_found > 0 and total_found % 100 == 0:
                logger.info(f"已获取并处理 {total_found} 篇论文...")
                
    except arxiv.UnexpectedEmptyPageError:
        # 将这个异常视为获取结束的正常信号，不做任何操作，让函数自然结束
        logger.info("捕获到 UnexpectedEmptyPageError，表明已成功到达结果流的末尾。")

    except Exception as e:
        logger.error(f"获取论文时发生未知错误: {e}", exc_info=True)
    
    logger.info(f"论文获取流程完成，共找到并返回了 {total_found} 篇论文。")


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
