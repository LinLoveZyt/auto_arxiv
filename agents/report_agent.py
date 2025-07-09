# agents/report_agent.py

import logging
import json
from typing import List, Dict, Any, Optional

from core import llm_client as llm_client_module
from core import config as config_module

logger = logging.getLogger(__name__)


REPORT_JSON_PROMPT_TEMPLATE = """
You are an expert AI researcher. Your task is to generate a structured JSON analysis for a given research paper.
The user will provide you with the paper's metadata, a high-quality summary, and a list of figures/tables with their surrounding textual context.
Based ONLY on the provided information, generate a JSON object with the following schema.

**JSON Schema:**
{{
  "title": "Paper Title",
  "authors": ["Author One", "Author Two"],
  "arxiv_id": "e.g., 2401.12345v1",
  "published_date": "YYYY-MM-DDTHH:MM:SSZ",
  "classification": {{
    "domain": "e.g., Computer Vision",
    "task": "e.g., Object Detection"
  }},
  "analysis": {{
    "problem_solved": "A concise, one-sentence description of the core problem the paper addresses. Derived from the summary.",
    "originality": "A one or two-sentence summary of the key innovation or unique contribution. What makes this paper novel? Derived from the summary.",
    "method_comparison": "Briefly describe how this method compares to or improves upon previous work mentioned in the summary."
  }},
  "images": {{
    "architecture_image": "The file path of the single image that best illustrates the overall system architecture, model structure, or workflow. Choose from the list below. If no single image is a clear winner, select the most comprehensive one.",
    "performance_image": "The file path of the single image that best showcases the main results, performance comparisons, or key quantitative findings (e.g., a chart with metrics, a table with scores). Choose from the list below. It must be different from the architecture image."
  }}
}}

**You must strictly follow these rules:**
1.  Generate ONLY the JSON object, with no other text or explanations before or after.
2.  Your entire response must be a single, valid JSON.
3.  The values for "architecture_image" and "performance_image" MUST be chosen from the file paths provided in the "Figures/Tables Context" section below.
4.  Do not invent new file paths. If no suitable image is found for a category, use an empty string "".

---
**Provided Information:**

**A. Paper Metadata:**
- Title: {title}
- Authors: {authors}
- arXiv ID: {arxiv_id}
- Published Date: {published_date}
- Current Classification: {classification}

**B. High-Quality Summary of the Paper:**
---
{paper_summary}
---

**C. Figures/Tables Context (for image selection):**
---
{media_summary_str}
---

**Based on all the information above, generate the JSON now.**
"""

def generate_report_json_for_paper(self, paper_meta: Dict[str, Any], report_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generates a structured JSON report for a single paper using a high-quality summary
        and contextual media list.

        Args:
            paper_meta: A dictionary containing the paper's metadata.
            report_context: A dictionary containing the 'paper_summary' and 'media_summary_str'.

        Returns:
            A dictionary representing the structured JSON report, or None on failure.
        """
        paper_title_short = paper_meta.get('title', 'Unknown Paper')[:50] + '...'
        logger.info(f"Starting to generate analytical report JSON for paper '{paper_title_short}'")

        try:
            # 数据已经由上游工作流准备好，我们直接使用
            paper_summary = report_context.get('paper_summary', 'Not available.')
            media_summary_str = report_context.get('media_summary_str', 'Not available.')

            # 注意：不再需要检查内容长度和截断的逻辑，因为输入已经很精简

            final_prompt = REPORT_JSON_PROMPT_TEMPLATE.format(
                title=paper_meta.get('title', ''),
                authors=', '.join(paper_meta.get('authors', [])),
                arxiv_id=paper_meta.get('arxiv_id', ''),
                published_date=paper_meta.get('published_date', ''),
                classification=paper_meta.get('classification', {}),
                paper_summary=paper_summary,
                media_summary_str=media_summary_str
            )

            report_json = self.llm_client.generate_json(
                prompt=final_prompt,
                system_message="You are a helpful AI assistant that specializes in scientific paper analysis and JSON generation."
            )

            if report_json:
                logger.info(f"✅ Successfully generated analytical report JSON for paper '{paper_title_short}'.")
                return report_json
            else:
                logger.error(f"❌ Failed to generate or parse JSON for paper '{paper_title_short}' after all retries.")
                return None

        except Exception as e:
            logger.error(f"An unexpected error occurred in generate_report_json_for_paper for '{paper_title_short}': {e}", exc_info=True)
            return None