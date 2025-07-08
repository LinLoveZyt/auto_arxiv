# workflows/query_flow.py

import logging
import arxiv
import json
import re
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple, Generator 

from core import config as config_module
from core import llm_client as llm_client_module
from hrag import embedding_engine as embedding_engine_module
from hrag import metadata_db, vector_db
from hrag import reranker as reranker_module

from data_ingestion.arxiv_fetcher import search_arxiv, parse_arxiv_result


# --- [修改] 导入正确的google搜索库 ---
try:
    from googlesearch import search as Google_Search_lib
except ImportError:
    logger.error("`googlesearch-python` library not found. Please install it using 'pip install googlesearch-python'")
    Google_Search_lib = None

logger = logging.getLogger(__name__)


def _extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """从一个URL中提取arXiv ID。例如: 'https://arxiv.org/abs/2401.12345v1?query=abc' -> '2401.12345'"""
    # 这个正则表达式现在只匹配数字、点号和可选的'v'及版本号
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([\d\.]+v?\d*)', url)
    if match:
        # 移除版本号信息
        return match.group(1).split('v')[0]
    return None


def _agent_generate_search_query(query_text: str) -> Dict[str, Any]:
    """根据用户的中文问题，生成适合Google搜索的英文关键词。"""
    today_str = date.today().isoformat()
    prompt = f"""
    You are a top-tier research assistant, skilled in web search. Your task is to analyze a user's Chinese query and convert it into a concise, effective English search query for Google.

    **User's Chinese Query:** "{query_text}"
    **Current Date:** {today_str}

    **Instructions:**
    1.  Identify the core concepts in the user's query.
    2.  Translate these concepts into standard English technical terms.
    3.  If the query implies a need for recent information (e.g., "最近", "latest", "trends", or mentions a recent date), add the current year to the search terms.
    4.  Combine them into a single, effective search string.

    **Output Format:** Your response MUST be a JSON object only.
    ```json
    {{
      "english_query": "Your generated English search query string"
    }}
    ```
    
    **Example:**
    User's Chinese Query: "介绍一下大语言模型多智能体系统最近的论文关注什么问题？"
    JSON Output:
    ```json
    {{
      "english_query": "large language model multi-agent system recent papers topics"
    }}
    ```
    """
    system_prompt = "You are a precise planning agent that outputs only JSON."
    plan = llm_client_module.llm_client.generate_json(prompt, system_prompt)

    if not plan or not plan.get("english_query"):
        logger.warning("Search query generation agent failed. Falling back to the original query.")
        return {"english_query": query_text}
    
    logger.info(f"Generated English search query: '{plan['english_query']}'")
    return plan

# workflows/query_flow.py

def _agent_filter_online_summaries(query: str, papers: List[Dict[str, Any]], max_papers: int) -> List[Dict[str, Any]]:
    if not papers: return []
    
    # 输入信息保持不变
    papers_info_str = "\n---\n".join([ f"ID: {p['arxiv_id']}\nTitle: {p['title']}\nSummary: {p.get('summary', 'N/A')[:1000]}" for p in papers ])
    
    # 使用极度简化的、命令式的Prompt
    prompt = f"""
    **User Query:**
    "{query}"

    **Candidate Papers:**
    ---
    {papers_info_str}
    ---

    **Instruction:**
    From the Candidate Papers, select up to {max_papers} papers most relevant to the User Query.
    Your output MUST be ONLY a JSON object with the key "promising_arxiv_ids".

    **JSON Output Format:**
    ```json
    {{
    "promising_arxiv_ids": ["id_of_paper_1", "id_of_paper_3"]
    }}
    ```
    """
    
    # 再次强化系统角色
    system_prompt = "You are a machine that only outputs JSON formatted according to the user's specified structure. Do not add any other text."

    result = llm_client_module.llm_client.generate_json(prompt, system_prompt)

    if result and 'promising_arxiv_ids' in result and isinstance(result['promising_arxiv_ids'], list):
        selected_ids = set(result['promising_arxiv_ids'])
        # 增加一步验证，确保筛选出的ID确实在原始论文列表中，防止模型幻觉
        original_ids = {p['arxiv_id'] for p in papers}
        valid_selected_ids = selected_ids.intersection(original_ids)

        selected_papers = [p for p in papers if p['arxiv_id'] in valid_selected_ids]
        logger.info(f"Online paper screener selected {len(selected_papers)} out of {len(papers)} papers.")
        return selected_papers
        
    logger.warning(f"Online paper screener failed to return valid JSON with 'promising_arxiv_ids' key. LLM result: {result}")
    return []

def _agent_synthesize_answer(query_text: str, context: str, source_type: str) -> Optional[str]:
    """
    根据提供的单一来源上下文，严格地合成答案。
    """
    prompt = f"""
    You are a meticulous AI research analyst. Your sole task is to answer the user's query based *strictly* on the "Provided Context" from the specified source. Do not use any external knowledge.

    **User's Query:** "{query_text}"

    **Provided Context from {source_type}:**
    ---
    {context if context.strip() else "No information was provided."}
    ---

    **Your Task:**
    1.  Carefully analyze the User's Query and the Provided Context.
    2.  Synthesize a direct, coherent answer using ONLY the information present in the Provided Context.
    3.  If the context does not contain relevant information, you MUST state this clearly. For example: "According to the searched materials, a direct answer to '{query_text}' could not be found. The available information covers the following topics: ..."
    4.  Be direct and confident in your summary of the provided materials.
    5.  DO NOT create a "References" or "参考文献" section.

    Now, write your synthesized answer.
    """
    system_prompt = "You are an assistant that synthesizes answers strictly from the provided text."
    return llm_client_module.llm_client.generate(prompt, system_prompt)




class QueryWorkflow:
    def __init__(self):
        self.config = config_module.get_current_config()
        self.rerank_top_n = self.config.get('MAX_RELEVANT_PAPERS', 5)
        logger.info(f"QueryWorkflow initialized with TOP_K_RESULTS={self.config['TOP_K_RESULTS']} and RERANK_TOP_N={self.rerank_top_n}")

    def _retrieve_local_context(self, query_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        logger.info("Starting retrieval from local H-RAG knowledge base...")
        query_vector = embedding_engine_module.embedding_engine.encode(query_text, prompt_name="query")
        if query_vector.size == 0: return "", []
        
        k_results = self.config['TOP_K_RESULTS']
        distances, retrieved_ids = vector_db.vector_db_manager.search(query_vector, k=k_results)
        if not retrieved_ids.any(): return "", []
        
        valid_ids = [int(id_val) for id_val in retrieved_ids if id_val >= 0]
        if not valid_ids: return "", []
        retrieved_metadata = metadata_db.get_metadata_for_ids(valid_ids)

        candidate_docs, doc_map = [], {}
        for meta_id in valid_ids:
            meta = retrieved_metadata.get(meta_id)
            if not meta or not meta.get('content_preview'): continue
            doc_content = meta['content_preview']
            candidate_docs.append(doc_content)
            doc_map[doc_content] = meta

        if not candidate_docs: return "", []

        logger.info(f"Retrieved {len(candidate_docs)} candidates. Starting reranking...")
        scores = reranker_module.reranker.rerank(query_text, candidate_docs)
        reranked_docs_with_scores = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

        logger.info(f"Reranking complete. Selecting top {self.rerank_top_n} papers.")
        top_reranked_metas, seen_source_ids = [], set()
        for doc_content, score in reranked_docs_with_scores:
            if len(top_reranked_metas) >= self.rerank_top_n: break
            meta = doc_map[doc_content]
            source_id = meta.get('source_id')
            if source_id and source_id not in seen_source_ids:
                top_reranked_metas.append(meta)
                seen_source_ids.add(source_id)
                logger.info(f"  - Selected paper (arXiv:{source_id}) with rerank score: {score:.4f}")

        context_parts, source_papers = [], []
        for meta in top_reranked_metas:
            source_id = meta['source_id']
            paper_details = metadata_db.get_paper_details_by_id(source_id)
            if paper_details:
                source_papers.append(paper_details)
                summary_text = paper_details.get('generated_summary') or paper_details.get('summary', 'N/A')
                context_parts.append(f"Source (arXiv:{source_id}): {paper_details.get('title', 'N/A')}\nSummary: {summary_text}")
        
        logger.info(f"Local retrieval after reranking found {len(source_papers)} highly relevant papers.")
        return "\n\n".join(context_parts), source_papers

    def _perform_online_search(self, query_text: str) -> Generator[Dict[str, Any], None, List[Dict[str, Any]]]:
        if not Google_Search_lib:
            yield {"type": "error", "message": "Google Search library is not installed. Cannot perform online search."}
            return []

        yield {"type": "progress", "message": "AI正在将您的问题转换为优化的英文搜索查询..."}
        plan = _agent_generate_search_query(query_text)
        
        google_query = f"{plan['english_query']} site:arxiv.org"
        yield {"type": "progress", "message": f"正在使用Google进行在线搜索: '{google_query}'"}
        
        try:
            search_result_urls = list(Google_Search_lib(google_query, num_results=15))
            if not search_result_urls:
                logger.warning("Google Search did not return any results.")
                return []

            arxiv_ids = [_extract_arxiv_id_from_url(url) for url in search_result_urls if _extract_arxiv_id_from_url(url)]
            unique_arxiv_ids = list(dict.fromkeys(arxiv_ids))
            if not unique_arxiv_ids:
                yield {"type": "progress", "message": "在线搜索未发现有效的arXiv论文链接。"}
                return []

            yield {"type": "progress", "message": f"找到了 {len(unique_arxiv_ids)} 篇候选论文，正在从arXiv获取摘要信息..."}
            search = arxiv.Search(id_list=unique_arxiv_ids, max_results=len(unique_arxiv_ids))
            online_papers = [parse_arxiv_result(res) for res in search.results()]

            if not online_papers:
                yield {"type": "progress", "message": "无法从arXiv获取候选论文的详细信息。"}
                return []

            yield {"type": "progress", "message": f"成功获取 {len(online_papers)} 篇论文的摘要，正在进行快速筛选..."}
            
            promising_online_papers = _agent_filter_online_summaries(
                query_text, 
                online_papers, 
                self.config['MAX_RELEVANT_PAPERS']
            )
            return promising_online_papers

        except Exception as e:
            logger.error(f"An error occurred during the online search process: {e}", exc_info=True)
            yield {"type": "error", "message": f"在线搜索时发生错误: {e}"}
            return []

    def run_stream(self, query_text: str, online_search_enabled: bool = False) -> Generator[Dict[str, Any], None, None]:
        logger.info(f"🚀 --- [Adaptive Query Workflow Started] Handling: '{query_text[:80]}' (Online: {online_search_enabled}) --- 🚀")
        
        final_papers, context_str, source_type = [], "", ""

        if online_search_enabled:
            source_type = "Online Search"
            # 注意：此处调用已移除 model_name
            online_search_generator = self._perform_online_search(query_text)
            final_result = []
            try:
                while True:
                    value = next(online_search_generator)
                    if isinstance(value, dict) and value.get("type") in ["progress", "error"]:
                        yield value
                    else:
                        final_result = value
            except StopIteration as e:
                final_result = e.value if e.value is not None else []
            final_papers = final_result
            context_str = "\n\n".join([f"Source (arXiv:{p['arxiv_id']}): {p['title']}\nPublished: {p['published_date'].strftime('%Y-%m-%d') if p.get('published_date') else 'N/A'}\nSummary: {p.get('summary', 'N/A')}" for p in final_papers])
        else:
            source_type = "Local Knowledge Base"
            yield {"type": "progress", "message": "正在检索本地知识库..."}
            context_str, final_papers = self._retrieve_local_context(query_text)

        if not final_papers:
            yield {"type": "final", "data": {"answer": f"非常抱歉，在{source_type}中未能找到与您问题相关的信息。", "sources": []}}
            return
            
        yield {"type": "progress", "message": f"已找到 {len(final_papers)} 篇相关论文，正在生成最终回答..."}
        
        # 调用时不再传递 model_name
        final_answer = _agent_synthesize_answer(query_text, context_str, source_type)
        
        if not final_answer:
            yield {"type": "final", "data": {"answer": "抱歉，我无法根据现有信息生成一个连贯的答案。可能信息之间存在矛盾或不足。", "sources": []}}
            return
            
        final_sources, seen_ids = [], set()
        for p in final_papers:
            arxiv_id = p.get("arxiv_id")
            if not arxiv_id or arxiv_id in seen_ids: continue
            
            source_details = {
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "summary": p.get("summary", ""),
                "authors": p.get("authors", [])
            }
            if 'generated_summary' in p:
                source_details["summary"] = p.get("generated_summary") or p.get("summary", "")
                source_details["pdf_url"] = f"/papers/pdf/{p.get('arxiv_id')}"
            else:
                source_details["pdf_url"] = p.get("pdf_url")
                
            final_sources.append(source_details)
            seen_ids.add(arxiv_id)

        yield {"type": "final", "data": {"answer": final_answer, "sources": final_sources}}