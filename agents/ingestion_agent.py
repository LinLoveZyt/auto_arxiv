# agents/ingestion_agent.py
import logging
import json
from typing import Dict, Any, Optional
import random
from core import config as config_module
from core import llm_client as llm_client_module

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


# agents/ingestion_agent.py

def propose_category_merges() -> Optional[Dict[str, Any]]:
    """
    使用LLM分析现有分类，提出合并建议，并过滤掉无效建议。
    新增了对大量分类进行采样的逻辑以防止性能问题。
    """
    if not llm_client_module.llm_client:
        logger.critical("LLM client is not initialized, cannot propose category merges.")
        return None

    logger.info("启动分类合并建议Agent...")
    known_categories = get_known_categories()
    
    total_tasks = sum(len(data.get("tasks", {})) for data in known_categories.values())
    
    if total_tasks < 2:
        logger.info("分类数量不足 (少于2个任务)，无需进行合并建议。")
        return {"proposals": []}

    # 关键修改：从全局配置中获取上限值
    current_config = config_module.get_current_config()
    MAX_TASKS_FOR_PROMPT = current_config.get('MAX_TASKS_FOR_PROMPT', 80)
    
    if total_tasks > MAX_TASKS_FOR_PROMPT:
        logger.warning(f"分类任务总数 ({total_tasks}) 超过了建议上限 ({MAX_TASKS_FOR_PROMPT})。将随机采样一部分分类进行分析，以避免性能问题。")
        
        all_tasks_flat = []
        for domain, data in known_categories.items():
            for task in data.get("tasks", {}).keys():
                all_tasks_flat.append((domain, task))
        
        sampled_tasks_flat = random.sample(all_tasks_flat, MAX_TASKS_FOR_PROMPT)
        
        sampled_categories = {}
        for domain, task in sampled_tasks_flat:
            if domain not in sampled_categories:
                sampled_categories[domain] = {"tasks": {}}
            sampled_categories[domain]["tasks"][task] = {}
        
        known_categories_for_prompt = sampled_categories
        logger.info(f"已随机采样 {MAX_TASKS_FOR_PROMPT} 个任务用于本次合并建议。")
    else:
        known_categories_for_prompt = known_categories

    known_categories_simple = {
        domain: list(data.get("tasks", {}).keys())
        for domain, data in known_categories_for_prompt.items()
    }
    
    prompt = MERGE_PROMPT_TEMPLATE.format(
        known_categories_str=json.dumps(known_categories_simple, indent=2, ensure_ascii=False)
    )
    system_prompt = "You are a precise, JSON-only output assistant specializing in taxonomy management."

    merge_proposal_result = llm_client_module.llm_client.generate_json(
        prompt=prompt,
        system_prompt=system_prompt
    )

    if not merge_proposal_result or "proposals" not in merge_proposal_result:
        logger.error("合并建议Agent未能生成有效的JSON。")
        return None

    raw_proposals = merge_proposal_result.get("proposals", [])
    valid_proposals = []
    for prop in raw_proposals:
        if not isinstance(prop, dict) or "from" not in prop or "to" not in prop:
            continue
        
        from_cat = prop.get("from")
        to_cat = prop.get("to")
        
        if not isinstance(from_cat, dict) or not isinstance(to_cat, dict):
            continue

        if from_cat != to_cat:
            valid_proposals.append(prop)
        else:
            logger.warning(f"已过滤掉一条无效的自我合并建议: {from_cat}")
            
    merge_proposal_result["proposals"] = valid_proposals

    logger.info(f"✅ 合并建议Agent完成，提出了 {len(valid_proposals)} 条有效建议。")
    return merge_proposal_result