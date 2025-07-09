# agents/report_agent.py

import logging
import json
from typing import List, Dict, Any, Optional

from core import llm_client as llm_client_module
from core import config as config_module

logger = logging.getLogger(__name__)


REPORT_JSON_PROMPT_TEMPLATE = """
You are an expert AI research analyst. Your task is to generate a professional, analytical report in a structured JSON format based on a high-quality AI-generated summary and a list of figures. Your most critical task is to select the two most representative images.
Input Information:

Paper Title: "{paper_title}"

Paper Authors: {paper_authors}

Paper Published Date: "{published_date}"

ArXiv ID: "{arxiv_id}"

Paper Classification: {classification_str}

A. AI-Generated Summary of the Paper (Base your analysis on this summary):
{paper_summary}

B. List of Available Figures (with their exact file paths and original captions):
{image_summary_str}

Your Instructions (MUST be followed precisely):
Deep Analysis: Based on the AI-Generated Summary (section A), generate a deep, analytical summary covering:
Problem Solved: What specific problem in the field does this paper address?
Originality & Innovation: What is the core novel idea or method proposed?
Methodology Comparison: Briefly compare the proposed method to existing approaches.
CRITICAL - Image Selection:
Review the caption for each image in section B.
Architecture Image: Find the image whose caption best describes the overall system architecture, framework, or workflow. Copy its exact path and place it in the architecture_image field of the output.
Performance Image: Find the image whose caption best shows experimental results, performance comparisons, or ablation studies (this could be a chart, graph, or table). Copy its exact path and place it in the performance_image field.
If a suitable image for a category is not available after reviewing all captions, you MUST use the JSON value null for that field.
JSON Output: Your response must be a single, valid JSON object only. Do not add comments or any other text outside the JSON structure.
Output JSON Structure:

{{
  "title": "The full title of the paper",
  "authors": ["Author One", "Author Two"],
  "arxiv_id": "The paper's arXiv ID",
  "published_date": "The paper's publication date",
  "classification": {{
      "domain": "The paper's domain",
      "task": "The paper's task"
  }},
  "analysis": {{
    "problem_solved": "A concise summary of the core problem addressed by the paper.",
    "originality": "A description of the paper's novel ideas and key innovations.",
    "method_comparison": "A brief comparison of the proposed method with existing alternatives."
  }},
  "images": {{
      "architecture_image": "images/figure1.jpg",
      "performance_image": null
  }}
}}
"""



# [新增] 为 qwen3 定制的、使用中文指令的、高效的 Prompt 模板
QWEN_REPORT_JSON_PROMPT_TEMPLATE_ZH = """
你是一位顶级的AI研究分析师。你的任务是根据提供的“AI生成的论文摘要”和“图表列表”，生成一份专业的、结构化的分析报告JSON。你最重要的任务是根据图表的标题，选择最能代表“模型架构”和“性能对比”的两张图片。

**输入信息:**
- **论文标题:** "{paper_title}"
- **论文作者:** {paper_authors}
- **发表日期:** "{published_date}"
- **ArXiv ID:** "{arxiv_id}"
- **AI生成的论文摘要 (请基于此摘要进行分析):**
---
{paper_summary}
---

- **可用的图表列表 (包含精确的文件路径和原始标题):**
---
{image_summary_str}
---

**你的指令 (必须严格遵守):**
1.  **深度分析:** 基于“AI生成的论文摘要”，生成深刻的分析内容，覆盖以下三点：
    -   **解决的问题 (problem_solved):** 这篇论文解决了该领域的什么具体问题？
    -   **独创性与创新点 (originality):** 论文提出的核心创新思想或方法是什么？
    -   **方法对比 (method_comparison):** 简要地将论文提出的方法与现有其他方法进行对比。
2.  **【最关键任务】图片选择:**
    -   仔细阅读“可用的图表列表”中每个图表的 `caption`（标题）。
    -   **架构图 (architecture_image):** 找出标题最能描述整个系统架构、框架或工作流程的图片。将其**精确的 `path`** 复制并填入输出的 `architecture_image` 字段。
    -   **性能图 (performance_image):** 找出标题最能展示实验结果、性能对比或消融研究的图表（这可能是一个图、一个表或一个流程图）。将其**精确的 `path`** 复制并填入输出的 `performance_image` 字段。
    -   如果审核完所有标题后，找不到适合某个类别的图片，你**必须**在该字段使用 JSON 的 `null` 值。
3.  **JSON 输出:** 你的回答必须是**一个完整的、合法的 JSON 对象**。不要在 JSON 结构之外添加任何注释、解释或其他文字。

**输出 JSON 结构:**
```json
{{
  "title": "论文的完整标题",
  "authors": ["作者一", "作者二"],
  "arxiv_id": "论文的ArXiv ID",
  "published_date": "论文的发表日期",
  "classification": {{
      "domain": "论文的领域",
      "task": "论文的任务"
  }},
  "analysis": {{
    "problem_solved": "对论文解决的核心问题的简明扼要的中文总结。",
    "originality": "对论文的创新思想和关键创新点的中文描述。",
    "method_comparison": "对所提出方法与现有替代方案的简要中文比较。"
  }},
  "images": {{
      "architecture_image": "images/figure1.jpg",
      "performance_image": null
  }}
}}
"""

def generate_report_json_for_paper(
    paper_meta: Dict[str, Any],
    structured_chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
    """
    Generates a structured JSON report for a single paper, including analysis and image selection.
    This version uses a high-quality AI-summary instead of full content to avoid context overflow
    and dynamically selects a prompt based on the current LLM.
    """
    if not llm_client_module.llm_client:
        logger.critical("LLM client is not initialized, cannot generate JSON for the report.")
        return None
        
    title = paper_meta.get("title", "N/A")
    arxiv_id = paper_meta.get("arxiv_id", "N/A")
    logger.info(f"Starting to generate analytical report JSON for paper '{title[:50]}...'")

    # ▼▼▼ [核心修改] 准备高效的上下文 ▼▼▼

    # 1. 获取AI生成的摘要，这是分析的基础
    paper_summary = paper_meta.get("generated_summary")
    if not paper_summary:
        logger.error(f"Paper {arxiv_id} is missing an AI-generated summary. Cannot generate report.")
        return None

    # 2. 预处理图片信息，确保图片列表完整
    image_summary_list = []
    for chunk in structured_chunks:
        if chunk.get("type") == "image" and chunk.get("img_path"):
            caption_text = " ".join(chunk.get("img_caption", [])).strip()
            if caption_text:
                relative_img_path = f"images/{chunk['img_path'].split('/')[-1]}"
                image_summary_list.append({
                    "path": relative_img_path,
                    "caption": caption_text
                })
    image_summary_str = json.dumps(image_summary_list, indent=2, ensure_ascii=False) if image_summary_list else "No images with captions found in the paper."

    # 3. 动态选择Prompt模板和系统提示
    current_config = config_module.get_current_config()
    target_model = current_config['OLLAMA_MODEL_NAME']
    
    # 默认为英文模板
    final_prompt_template = REPORT_JSON_PROMPT_TEMPLATE
    system_prompt = "You are an assistant that only outputs strictly formatted JSON."
    
    # 如果是qwen3模型，切换为中文模板
    if "qwen3" in target_model.lower():
        final_prompt_template = QWEN_REPORT_JSON_PROMPT_TEMPLATE_ZH
        # system_prompt 可以在 llm_client 中被自动加强，这里保持通用即可
        logger.info(f"Detected model '{target_model}', switching to specialized Chinese prompt for report generation.")

    # 4. 准备其他Prompt变量
    classification_data = paper_meta.get('classification_result', {"domain": "N/A", "task": "N/A"})
    
    # 5. 格式化最终的Prompt
    final_prompt = final_prompt_template.format(
        paper_title=title,
        paper_authors=json.dumps(paper_meta.get("authors", []), ensure_ascii=False),
        published_date=str(paper_meta.get("published_date", "N/A")).split('T')[0], # 修正日期格式
        arxiv_id=arxiv_id,
        classification_str=json.dumps(classification_data, ensure_ascii=False),
        paper_summary=paper_summary, # 使用AI摘要
        image_summary_str=image_summary_str, # 使用完整的图片列表
    )
    # ▲▲▲ 上下文准备结束 ▲▲▲

    report_json = llm_client_module.llm_client.generate_json(
        prompt=final_prompt,
        system_prompt=system_prompt
    )

    if not report_json:
        logger.error(f"LLM failed to generate valid report JSON for paper '{title[:50]}...'.")
        return None

    # 增强验证，确保关键字段存在
    required_keys = ["title", "authors", "arxiv_id", "published_date", "classification", "analysis", "images"]
    if not all(key in report_json for key in required_keys):
        logger.error(f"The JSON returned by the LLM is missing one or more required keys. JSON: {report_json}")
        return None
        
    # 确保返回的 classification 字段被正确填充，以供 pdf_generator 使用
    if not report_json.get('classification'):
        report_json['classification'] = classification_data
    
    logger.info(f"✅ Successfully generated analytical report JSON for paper '{title[:50]}...'.")
    return report_json