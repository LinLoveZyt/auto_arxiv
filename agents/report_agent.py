# agents/report_agent.py

import logging
import json
from typing import List, Dict, Any, Optional

from core import llm_client as llm_client_module
from core import config as config_module

logger = logging.getLogger(__name__)


# ▼▼▼ [核心修改] 模板被大幅强化，指令更明确 ▼▼▼
REPORT_JSON_PROMPT_TEMPLATE = """
You are an expert AI research analyst. Your task is to generate a professional, analytical report in a structured JSON format. Your most critical task is to select the two most representative images based on a provided summary of figures.

**Input Information:**
- **Paper Title:** "{paper_title}"
- **Paper Authors:** {paper_authors}
- **Paper Published Date:** "{published_date}"
- **Paper ArXiv ID:** "{arxiv_id}"
- **Paper Classification:** {classification_str}

**A. Summary of Available Figures (with their exact file paths and original captions):**
---
{image_summary_str}
---

**B. Full Structured Content of the Paper (for context):**
---
{structured_content}
---

**Your Instructions (MUST be followed precisely):**
1.  **Deep Analysis:** Based on all the provided text (section B), generate a deep, analytical summary covering:
    -   **Problem Solved:** What specific problem in the field does this paper address?
    -   **Originality & Innovation:** What is the core novel idea or method proposed?
    -   **Methodology Comparison:** Briefly compare the proposed method to existing approaches.
2.  **CRITICAL - Image Selection:**
    -   Review the `caption` for each image in **section A**.
    -   **Architecture Image:** Find the image whose caption best describes the overall system architecture, framework, or workflow. **Copy its exact `path`** and place it in the `architecture_image` field of the output.
    -   **Performance Image:** Find the image whose caption best shows experimental results, performance comparisons, or ablation studies (this could be a chart, graph, or table). **Copy its exact `path`** and place it in the `performance_image` field.
    -   If a suitable image for a category is not available after reviewing all captions, you **MUST** use the JSON value `null` for that field.
3.  **JSON Output:** Your response must be a single, valid JSON object only, strictly adhering to the specified format. **Do not add comments or any other text outside the JSON structure.**

**Output JSON Structure:**
```json
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
```
"""

def generate_report_json_for_paper(
    paper_meta: Dict[str, Any],
    structured_chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
    """
    Generates a structured JSON report for a single paper, including analysis and image selection.
    """
    if not llm_client_module.llm_client:
        logger.critical("LLM client is not initialized, cannot generate JSON for the report.")
        return None
        
    title = paper_meta.get("title", "N/A")
    arxiv_id = paper_meta.get("arxiv_id", "N/A")
    logger.info(f"Starting to generate analytical report JSON for paper '{title[:50]}...'")

    # 1. 预处理图片信息
    image_summary_list = []
    for chunk in structured_chunks:
        if chunk.get("type") == "image" and chunk.get("img_path"):
            caption_text = " ".join(chunk.get("img_caption", [])).strip()
            if caption_text:
                # 关键修改：存储的路径现在是相对路径
                relative_img_path = f"images/{chunk['img_path'].split('/')[-1]}"
                image_summary_list.append({
                    "path": relative_img_path,
                    "caption": caption_text
                })

    image_summary_str = json.dumps(image_summary_list, indent=2, ensure_ascii=False) if image_summary_list else "No images with captions found in the paper."

    # 2. 准备其他Prompt变量
    content_str = json.dumps(structured_chunks, indent=2, ensure_ascii=False)

    current_config = config_module.get_current_config()
    MAX_CONTENT_CHARS = current_config.get('MAX_CONTENT_FOR_REPORT', 15000)

    if len(content_str) > MAX_CONTENT_CHARS:
        original_len = len(content_str)
        content_str = content_str[:MAX_CONTENT_CHARS] + "\n... (content truncated for brevity)"
        logger.warning(f"Paper content for {arxiv_id} is too long ({original_len} chars). Truncated to {MAX_CONTENT_CHARS} for report generation prompt.")

    classification_data = paper_meta.get('classification_result', {"domain": "N/A", "task": "N/A"})

    final_prompt = REPORT_JSON_PROMPT_TEMPLATE.format(
        paper_title=title,
        paper_authors=json.dumps(paper_meta.get("authors", []), ensure_ascii=False),
        published_date=str(paper_meta.get("published_date", "N/A")),
        arxiv_id=arxiv_id,
        classification_str=json.dumps(classification_data, ensure_ascii=False),
        image_summary_str=image_summary_str,
        structured_content=content_str
    )

    system_prompt = "You are an assistant that only outputs strictly formatted JSON."

    report_json = llm_client_module.llm_client.generate_json(
        prompt=final_prompt,
        system_prompt=system_prompt
    )

    if not report_json:
        logger.error(f"LLM failed to generate valid report JSON for paper '{title[:50]}...'.")
        return None

    if 'analysis' not in report_json or 'classification' not in report_json:
        logger.error(f"The JSON returned by the LLM is missing required fields (analysis/classification): {report_json}")
        return None
        
    if 'images' not in report_json or not isinstance(report_json.get('images'), dict):
        logger.warning(f"The JSON returned by the LLM is missing a valid 'images' dictionary. Report will not have images. JSON: {report_json}")
        report_json['images'] = {}

    logger.info(f"✅ Successfully generated analytical report JSON for paper '{title[:50]}...'.")
    return report_json