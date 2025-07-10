# agents/ingestion_agent.py
import logging
import json
from typing import Dict, Any, Optional
import random
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from core import config as config_module
from core import llm_client as llm_client_module
from hrag import metadata_db, embedding_engine, reranker as reranker_module


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
    You are a top-tier AI research paper classification expert. Your task is to accurately categorize a paper into a "domain" and a "task" based on its title and abstract.

    **Paper to Classify:**
    - **Title:** "{paper_title}"
    - **Abstract:** "{paper_abstract}"

    **Your Instructions:**
    1.  Carefully read the paper's title and abstract to understand its core research topic.
    2.  Based on the content, determine the most appropriate high-level research **domain** (e.g., "Computer Vision", "Natural Language Processing", "Robotics", "Reinforcement Learning").
    3.  Then, determine the specific **task** or sub-field within that domain (e.g., "Object Detection", "Machine Translation", "Motion Planning", "Q-Learning").
    4.  Your response must be a JSON object only. Do not include any explanations, comments, or other extraneous text.

    **Output Format (JSON):**
    ```json
    {{
    "domain": "Domain Name",
    "task": "Task Name"
    }}
    ```
    """



# 用于在一个已确认的冗余簇中，选举出最规范的名称
ARBITRATION_PROMPT_TEMPLATE = """
You are a world-class AI research classification taxonomist. From the following list of synonymous research categories, your task is to select the single best one to serve as the canonical name.

List of Synonymous Categories:
{category_list_str}
Instructions:

The canonical name should be the most standard, widely-accepted, professional, and complete term. Avoid abbreviations or colloquialisms.

Your response MUST be a single JSON object containing the chosen domain and task.

Output JSON Structure:

JSON

{{
  "canonical_form": {{
    "domain": "The chosen standard domain name",
    "task": "The chosen standard task name"
  }}
}}
"""

SUBSET_PROMPT_TEMPLATE = """
You are an expert AI research taxonomist. Your task is to analyze a list of research categories and identify one or more groups of true synonyms within it.

**List of Categories to Analyze:**
---
{category_list_str}
---

**Instructions:**
1.  Read the entire list to understand the topics.
2.  Identify groups (subsets) where all members are **semantically equivalent**. They must be true synonyms, abbreviations, or different phrasings for the exact same concept.
3.  A category should only appear in one group.
4.  If a category has no synonyms in the list, do not include it in any group.
5.  Your output MUST be a single JSON object containing a list of these groups. If no synonym groups are found, return an empty list.

**Output JSON Structure:**
```json
{{
  "synonym_groups": [
    [
      {{"domain": "CV", "task": "Object Detection"}},
      {{"domain": "Computer Vision", "task": "Object Detection"}}
    ],
    [
      {{"domain": "NLP", "task": "Text Generation"}},
      {{"domain": "Natural Language Processing", "task": "Text Generation"}}
    ]
  ]
}}
"""

RAG_CONTEXT_CLASSIFICATION_PROMPT_TEMPLATE = """
You are a world-class AI research classification taxonomist. Your task is to determine the most accurate and consistent classification for a new paper by leveraging context from highly similar existing papers.

**New Paper to Classify:**
- **Title:** "{new_paper_title}"
- **Abstract:** "{new_paper_abstract}"

**Your Initial Independent Suggestion:**
- **Domain:** "{candidate_domain}"
- **Task:** "{candidate_task}"

**Reference Context (Most Similar Papers from Knowledge Base):**
---
{similar_papers_context}
---

**Your Task & Decision Process:**
1.  **Analyze Context:** Carefully review the new paper's information and compare it against the reference papers. Pay close attention to their titles, abstracts, existing classifications, and rerank scores (a higher score means more relevant).
2.  **Make a Decision:**
    -   **Strong Alignment:** If the new paper's topic is a clear synonym or identical to one of the high-score reference papers, you **MUST** adopt the **existing classification** from that reference paper to maintain consistency.
    -   **Novel Contribution:** If the new paper introduces a genuinely new concept not covered by the references, and your initial suggestion is accurate, you should use your **initial suggestion**.
    -   **Refined Novelty:** If your initial suggestion is close but the context suggests a better phrasing or a more precise new category, feel free to refine it into a **new, improved classification**.
3.  **Provide Reasoning:** Briefly explain your decision in one sentence.
4.  **Final Output:** Your response MUST be a single, valid JSON object, adhering strictly to the specified structure. Do not add any text outside the JSON.

**Output JSON Structure:**
```json
{{
  "reasoning": "A brief explanation of your decision-making process.",
  "final_classification": {{
    "domain": "The final, most appropriate domain name",
    "task": "The final, most appropriate task name"
  }}
}}
"""


def get_known_categories() -> Dict:
    """从JSON文件加载已知的分类体系。"""
    try:
        if config_module.CATEGORIES_JSON_PATH.exists():
            with open(config_module.CATEGORIES_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"无法读取或解析分类文件: {e}")
    return {}

def _update_known_categories(domain: str, task: str):
    """将新的分类更新到JSON文件中。"""
    categories = get_known_categories()
    
    # 确保Domain存在
    if domain not in categories:
        categories[domain] = {"tasks": {}}
    
    # 确保Task存在
    if task not in categories[domain]["tasks"]:
        categories[domain]["tasks"][task] = {} # 可以预留字段给未来，如description

    try:
        with open(config_module.CATEGORIES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=4, ensure_ascii=False)
        logger.info(f"分类体系已更新: Domain='{domain}', Task='{task}'")
    except IOError as e:
        logger.error(f"无法写入分类文件: {e}")

def classify_paper(title: str, abstract: str) -> Optional[Dict[str, Any]]:
    """使用LLM对论文进行分类，并动态更新全局分类文件。"""
    if not llm_client_module.llm_client:
        logger.critical("LLM client is not initialized, cannot perform paper classification.")
        return None

    logger.info(f"开始智能分类: '{title[:50]}...'")

    final_prompt = PROMPT_TEMPLATE.format(
        paper_title=title,
        paper_abstract=abstract
    )

    system_prompt = "You are a precise, JSON-only output assistant specializing in paper classification."
    
    classification_result = llm_client_module.llm_client.generate_json(
        prompt=final_prompt,
        system_prompt=system_prompt
    )

    if not classification_result:
        logger.error(f"LLM未能为论文 '{title[:50]}...' 生成有效的分类JSON。")
        return None

    required_keys = ["domain", "task"]
    if not all(key in classification_result and classification_result[key] for key in required_keys):
        logger.error(f"LLM返回的JSON缺少必要字段或字段为空: {classification_result}")
        return None

    domain = classification_result["domain"]
    task = classification_result["task"]
    
    _update_known_categories(domain, task)
    
    logger.info(f"✅ 论文 '{title[:50]}...' 分类成功: "
                f"Domain='{domain}', Task='{task}'")

    return classification_result

ALIGNMENT_PROMPT_TEMPLATE = """
    You are a meticulous AI research classification expert. Your task is to align a newly classified paper category with an existing list of categories to maintain consistency.

    **Existing Category Structure:**
    (A JSON object where keys are high-level domains and values are lists of specific tasks within that domain)
    ---
    {known_categories_str}
    ---

    **Newly Classified Category:**
    - **Domain:** "{new_domain}"
    - **Task:** "{new_task}"

    **Your Instructions:**
    1.  **Analyze the New Category:** Examine the "Newly Classified Category".
    2.  **Compare with Existing List:** Check if the new (domain, task) pair is a synonym, a more specific/general version, or an obvious alias of any category in the "Existing Category Structure".
        -   **Example 1 (Synonym):** If new is ("Large Language Models", "Instruction Following") and existing has ("Natural Language Processing", "Instruction Tuning"), you should align them.
        -   **Example 2 (Alias):** If new is ("CV", "Object Detection") and existing has ("Computer Vision", "Object Detection"), you should align them to the more formal "Computer Vision".
    3.  **Determine Final Category:**
        -   If you find a clear match, use the **EXISTING** domain and task names for the final output.
        -   If the new category is genuinely novel and does not fit any existing category, use the **NEW** domain and task names as the final output.
    4.  **JSON Output:** Your response must be a JSON object only, strictly adhering to the specified format.

    **Output JSON Structure:**
    ```json
    {{
      "final_domain": "The aligned or original domain name",
      "final_task": "The aligned or original task name"
    }}
    ```
    """


def align_classification(
    raw_classification: Dict[str, str],
    known_categories: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    使用LLM将一个原始分类结果与已知分类体系对齐。
    """
    if not llm_client_module.llm_client:
        logger.critical("LLM client is not initialized, cannot perform classification alignment.")
        return None

    new_domain = raw_classification["domain"]
    new_task = raw_classification["task"]

    # 如果还没有任何已知分类，直接返回符合期望格式的结果
    if not known_categories:
        return {"final_domain": new_domain, "final_task": new_task}

    # 检查新分类是否已经存在于已知分类中
    if new_domain in known_categories and new_task in known_categories[new_domain].get("tasks", {}):
        logger.info(f"新分类 ('{new_domain}', '{new_task}') 已是标准格式，无需对齐。")
        # vvv [修改] 即使无需对齐，也要返回正确的键名 vvv
        return {"final_domain": new_domain, "final_task": new_task}
        # ^^^ [修改] ^^^

    logger.info(f"开始对新分类 ('{new_domain}', '{new_task}') 进行对齐...")
    known_categories_simple = {
        domain: list(data.get("tasks", {}).keys())
        for domain, data in known_categories.items()
    }
    
    final_prompt = ALIGNMENT_PROMPT_TEMPLATE.format(
        known_categories_str=json.dumps(known_categories_simple, indent=2, ensure_ascii=False),
        new_domain=new_domain,
        new_task=new_task
    )

    system_prompt = "You are a precise, JSON-only output assistant specializing in category alignment."
    
    aligned_result = llm_client_module.llm_client.generate_json(
        prompt=final_prompt,
        system_prompt=system_prompt
    )

    # vvv [修改] 简化逻辑，直接返回LLM的结果或原始分类（使用正确的键） vvv
    if aligned_result and "final_domain" in aligned_result and "final_task" in aligned_result:
        logger.info(f"✅ 对齐完成: ('{new_domain}', '{new_task}') -> ('{aligned_result['final_domain']}', '{aligned_result['final_task']}')")
        # 直接返回LLM给出的、包含正确键的JSON
        return aligned_result
    
    logger.error(f"对齐Agent未能生成有效JSON。将使用原始分类。原始: {raw_classification}")
    # 对齐失败时，返回原始分类，但键名要符合约定
    return {"final_domain": new_domain, "final_task": new_task}
    # ^^^ [修改] ^^^


MERGE_PROMPT_TEMPLATE = """
You are a world-class AI research classification taxonomist. Your goal is to consolidate a given list of research categories into a clean, canonical, and non-redundant hierarchy.

**Current Category List (JSON):**
---
{known_categories_str}
---

**Your Task & Rules:**
1.  **Goal is Consolidation:** Your primary goal is to REDUCE the number of categories by merging redundant ones.
2.  **Identify Merge Candidates:** Scan the list for all types of redundancies:
    -   **Abbreviations:** e.g., "CV" should be merged into "Computer Vision".
    -   **Synonyms & Aliases:** e.g., "Instruction Following" should be merged into "Instruction Tuning" if they represent the same concept in the field.
    -   **Formatting & Phrasing:** e.g., "Object-Detection" into "Object Detection"; "Large Language Model" into "Large Language Models".
3.  **Determine Canonical Form:** For each redundant pair, the "to" category MUST be the most standard, widely-accepted, or complete term.
4.  **CRITICAL RULE: "FROM" MUST EXIST.** The "from" category in any proposal **MUST** be a category that exists in the "Current Category List" provided above. Do not invent "from" categories that are not in the list.
5.  **CRITICAL RULE: NO SELF-MERGING.** You MUST NOT propose merging a category with itself. The "from" and "to" categories in any proposal must be different. Any proposal where "from" and "to" are identical is invalid.
6.  **Provide Reason:** For each proposal, provide a short, clear reason: "Abbreviation", "Synonym", "Typo/Formatting", or "More Common Term".
7.  **Strict JSON Output:** You MUST output a JSON object containing a list of valid proposals. If no merges are needed, return an empty list (`"proposals": []`).

**Output JSON Structure:**
```json
{{
  "proposals": [
    {{
      "from": {{"domain": "CV", "task": "Object Detection"}},
      "to": {{"domain": "Computer Vision", "task": "Object Detection"}},
      "reason": "Abbreviation"
    }}
  ]
}}
```
"""



def propose_category_merges() -> Optional[Dict[str, Any]]:
    """
    使用“聚类-识别子集-仲裁”的三阶段先进算法，提出分类合并建议。
    V3版 (用户启发版):
    1.  宽容地聚类，得到包含大量相关项的“候选簇”。
    2.  [新增] 让LLM审视每个大候选簇，并从中提取一个或多个纯净的“同义词子集”。
    3.  对每个提取出的高质量子集，再由LLM仲裁选出规范名称。
    """
    if not all([llm_client_module.llm_client, embedding_engine.embedding_engine]):
        logger.critical("LLM client or Embedding Engine not initialized, cannot propose merges.")
        return None

    logger.info("启动新版分类合并建议Agent v3 (聚类-子集发现-仲裁)...")
    
    # 1. 从数据库获取所有分类并进行向量化 (与V2版相同)
    all_db_categories = metadata_db.get_all_domains_and_tasks()
    flat_categories = []
    for domain, tasks in all_db_categories.items():
        for task in tasks:
            flat_categories.append({"domain": domain, "task": task})

    if len(flat_categories) < 2:
        logger.info("数据库中的分类总数不足 (少于2个)，无需进行合并建议。")
        return {"proposals": []}

    logger.info(f"正在为从数据库获取的全部 {len(flat_categories)} 个分类生成嵌入向量...")
    descriptions = [f"Research Domain: {cat['domain']}, Specific Task: {cat['task']}" for cat in flat_categories]
    try:
        vectors = embedding_engine.embedding_engine.encode(descriptions)
    except Exception as e:
        logger.error(f"为分类生成嵌入向量时失败: {e}", exc_info=True)
        return None

    # 2. 使用层次聚类，得到大的“候选簇” (与V2版相同)
    current_config = config_module.get_current_config()
    threshold = current_config.get("CATEGORY_CLUSTER_THRESHOLD", 0.3)
    logger.info(f"开始对向量进行层次聚类，距离阈值(Threshold)为: {threshold}")
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric='cosine', linkage='average')
    labels = clustering.fit_predict(vectors)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters: clusters[label] = []
        clusters[label].append(flat_categories[i])
    
    candidate_clusters = [c for c in clusters.values() if len(c) > 1]
    logger.info(f"聚类完成，发现 {len(candidate_clusters)} 个大的候选簇。")

    # 3. 遍历大簇，让LLM提取纯净的“同义词子集”并进行仲裁
    final_proposals = []
    system_prompt_json = "You are a precise, JSON-only output assistant."

    for i, cluster in enumerate(candidate_clusters):
        logger.info(f"--- 正在处理候选簇 {i+1}/{len(candidate_clusters)} ---")
        
        # 步骤 3a: LLM 识别同义词子集
        cluster_json_str = json.dumps(cluster, indent=2, ensure_ascii=False)
        subset_prompt = SUBSET_PROMPT_TEMPLATE.format(category_list_str=cluster_json_str)
        subset_result = llm_client_module.llm_client.generate_json(subset_prompt, system_prompt_json)

        if not subset_result or "synonym_groups" not in subset_result or not subset_result["synonym_groups"]:
            logger.info("LLM 在此大簇中未发现任何同义词子集，已跳过。")
            continue
        
        synonym_groups = subset_result["synonym_groups"]
        logger.info(f"LLM 发现了 {len(synonym_groups)} 个同义词子集，开始逐一进行仲裁...")

        # 步骤 3b: 对每个子集进行仲裁
        for j, group in enumerate(synonym_groups):
            if not isinstance(group, list) or len(group) < 2: continue # 跳过无效或单个成员的组
            
            logger.info(f"  -- 正在仲裁子集 {j+1}/{len(synonym_groups)} --")
            group_str_list = [f'- {g["domain"]} > {g["task"]}' for g in group]
            arbitration_prompt = ARBITRATION_PROMPT_TEMPLATE.format(category_list_str="\n".join(group_str_list))
            arbitration_result = llm_client_module.llm_client.generate_json(arbitration_prompt, system_prompt_json)

            if not arbitration_result or "canonical_form" not in arbitration_result:
                logger.error(f"LLM 仲裁失败，无法为子集选举出规范名称。已跳过。子集内容: {group_str_list}")
                continue

            canonical_form = arbitration_result["canonical_form"]
            if canonical_form not in group:
                logger.warning(f"LLM选举出了一个不在子集内的规范名称: {canonical_form}。已跳过此子集。")
                continue

            logger.info(f"  选举完成！规范名称为: {canonical_form['domain']} > {canonical_form['task']}")

            # 步骤 3c: 为当前子集生成合并计划
            for category in group:
                if category != canonical_form:
                    proposal = {
                        "from": category,
                        "to": canonical_form,
                        "reason": "AI-Identified Synonym Group"
                    }
                    final_proposals.append(proposal)

    logger.info(f"✅ 合并建议流程完成，共生成 {len(final_proposals)} 条高质量、无冲突的合并建议。")
    return {"proposals": final_proposals}

def classify_paper_with_rag_context(title: str, abstract: str) -> Optional[Dict[str, str]]:
    """
    使用先进的“独立分类 + RAG辅助对齐”两阶段流程对论文进行分类。
    """
    logger.info(f"--- [高级分类流程启动] 论文: '{title[:50]}...' ---")

    # 阶段一：获取独立的候选分类
    logger.info("阶段 1/3: 生成独立的候选分类...")
    candidate_classification = classify_paper(title, abstract)
    if not candidate_classification:
        logger.error("无法生成候选分类，高级分类流程中止。")
        return None
    logger.info(f"候选分类为: {candidate_classification}")

    # 阶段二：基于候选分类，进行分层RAG检索
    logger.info("阶段 2/3: 基于候选分类进行分层RAG检索...")
    candidate_domain = candidate_classification['domain']
    candidate_task = candidate_classification['task']

    # 2a. 从数据库获取同分类的论文
    similar_papers_from_db = metadata_db.get_papers_by_classification(candidate_domain, candidate_task)

    if not similar_papers_from_db:
        logger.warning(f"在分类 '{candidate_domain}/{candidate_task}' 下未找到任何参考论文。将直接采纳候选分类。")
        # 直接更新全局分类文件并返回
        _update_known_categories(candidate_domain, candidate_task)
        return candidate_classification

    # 2b. 准备重排数据
    documents_to_rerank = []
    doc_map = {}
    for paper in similar_papers_from_db:
        # 使用AI生成的摘要（如果存在）或原始摘要
        summary = paper.get('generated_summary') or paper.get('summary', '')
        if summary:
            documents_to_rerank.append(summary)
            doc_map[summary] = paper

    if not documents_to_rerank:
        logger.warning("参考论文缺少有效摘要，无法进行重排。将直接采纳候选分类。")
        _update_known_categories(candidate_domain, candidate_task)
        return candidate_classification

    # 2c. 执行重排
    logger.info(f"正在对 {len(documents_to_rerank)} 篇参考论文的摘要进行重排...")
    try:
        # 使用新论文的摘要作为查询来找最相似的已有摘要
        scores = reranker_module.reranker.rerank(abstract, documents_to_rerank)
        reranked_docs_with_scores = sorted(zip(documents_to_rerank, scores), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"Reranker执行失败: {e}", exc_info=True)
        # 失败时，同样采纳候选分类作为降级策略
        _update_known_categories(candidate_domain, candidate_task)
        return candidate_classification

    # 2d. 筛选上下文并格式化
    top_references = []
    # 筛选分数大于0.5的，最多5篇
    for doc_content, score in reranked_docs_with_scores:
        if len(top_references) >= 5: break
        if score > 0.5:
            paper_meta = doc_map[doc_content]
            # 获取这篇参考论文自己的分类
            ref_classification = metadata_db.get_paper_details_by_id(paper_meta['arxiv_id']).get('classification_result', {'domain': candidate_domain, 'task': candidate_task})

            top_references.append({
                "title": paper_meta['title'],
                "abstract": doc_content[:500] + '...', # 摘要截断以防prompt过长
                "classification": ref_classification,
                "score": score
            })

    if not top_references:
        logger.info("没有找到强相关（分数>0.5）的参考论文。将直接采纳候选分类。")
        _update_known_categories(candidate_domain, candidate_task)
        return candidate_classification

    similar_papers_context = "\n---\n".join([
        f"[Reference Paper | Rerank Score: {ref['score']:.4f}]\n"
        f"- Title: {ref['title']}\n"
        f"- Abstract Snippet: {ref['abstract']}\n"
        f"- Existing Classification: Domain='{ref['classification']['domain']}', Task='{ref['classification']['task']}'"
        for ref in top_references
    ])

    # 阶段三：调用LLM进行最终决策
    logger.info("阶段 3/3: 调用LLM进行最终决策...")
    final_prompt = RAG_CONTEXT_CLASSIFICATION_PROMPT_TEMPLATE.format(
        new_paper_title=title,
        new_paper_abstract=abstract,
        candidate_domain=candidate_domain,
        candidate_task=candidate_task,
        similar_papers_context=similar_papers_context
    )

    system_prompt = "You are a precise, JSON-only output assistant specializing in paper classification."
    llm_result = llm_client_module.llm_client.generate_json(final_prompt, system_prompt)

    if llm_result and "final_classification" in llm_result:
        final_classification = llm_result["final_classification"]
        reasoning = llm_result.get("reasoning", "N/A")
        logger.info(f"✅ 高级分类流程完成。决策理由: {reasoning}")
        logger.info(f"最终分类为: {final_classification}")
        # 将最终确定的、高质量的分类更新到全局分类文件
        _update_known_categories(final_classification["domain"], final_classification["task"])
        return final_classification
    else:
        logger.error(f"LLM未能对分类做出最终决策。将采纳候选分类作为降级方案。LLM原始返回: {llm_result}")
        _update_known_categories(candidate_domain, candidate_task)
        return candidate_classification

def export_categories_to_json() -> bool:
    """
    从数据库中查询所有分类，并覆盖写入到 categories.json 文件。
    这是确保“数据库为唯一事实来源”的关键函数。
    """
    logger.info("正在从数据库导出分类体系到 categories.json...")
    try:
        # 从数据库获取所有分类
        db_categories = metadata_db.get_all_domains_and_tasks()

        # 构造成 categories.json 所需的格式
        output_data = {}
        for domain, tasks in db_categories.items():
            output_data[domain] = {"tasks": {task: {} for task in tasks}}

        # 写入文件
        with open(config_module.CATEGORIES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ 分类体系已成功同步到 {config_module.CATEGORIES_JSON_PATH}")
        return True
    except Exception as e:
        logger.error(f"导出分类到JSON时失败: {e}", exc_info=True)
        return False