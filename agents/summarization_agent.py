# agents/summarization_agent.py

import logging
from typing import List, Dict, Any, Optional

from core import llm_client as llm_client_module

logger = logging.getLogger(__name__)

# ▼▼▼ [修改] 此函数现在适配MonkeyOCR的输出格式 ▼▼▼
def _clean_and_prepare_content(structured_chunks: List[Dict[str, Any]]) -> str:
    """
    Cleans text blocks parsed from MonkeyOCR and concatenates them into coherent text.
    """
    logger.info("Starting to clean and prepare paper content from MonkeyOCR output...")
    
    # ▼▼▼ 核心修改：更新有效类型以匹配MonkeyOCR的输出 ▼▼▼
    # 在MonkeyOCR的输出中，所有文本块的类型都是 'text'
    valid_types = {"text"}
    
    content_parts = []
    for chunk in structured_chunks:
        chunk_type = chunk.get("type")
        
        # 我们只关心文本块，忽略图片、表格等用于摘要
        if chunk_type not in valid_types:
            continue
        
        text = chunk.get("text", "").strip()
        
        # 保留一些基本的清洗规则
        if len(text) < 20 and not text.endswith('.'):
             continue
        if text.lower().startswith("preprint.") or text.lower().startswith("figure"):
            continue

        content_parts.append(text)
        
    logger.info(f"Content cleaning complete, {len(content_parts)} valid text blocks retained.")
    return "\n\n".join(content_parts)


def _summarize_chunk(chunk_text: str) -> str:
    """Uses an LLM to summarize a single text chunk."""
    if not llm_client_module.llm_client:
        return "Error: LLM client not initialized."
        
    prompt = f"""
    The following is a section of a research paper. Please summarize the core ideas of this section in 1-2 concise sentences.

    ---
    {chunk_text}
    ---

    Core Ideas Summary:
    """
    system_prompt = "You are a research paper summarization assistant, skilled in scientific literature."
    
    summary = llm_client_module.llm_client.generate(prompt, system_prompt)
    return summary.strip() if summary else ""


def summarize_paper_from_chunks(
    structured_chunks: List[Dict[str, Any]],
    paper_title: str
) -> Optional[str]:
    """
    Generates a comprehensive summary from parsed text chunks using a Map-Reduce process.
    """
    if not llm_client_module.llm_client:
        logger.critical("LLM client is not initialized, cannot generate paper summary.")
        return None

    logger.info(f"--- Map-Reduce Summary Process Started: Paper '{paper_title[:50]}...' ---")

    # 0. Clean and prepare input content
    full_content = _clean_and_prepare_content(structured_chunks)
    if not full_content:
        logger.error("Content is empty after cleaning, cannot generate summary.")
        return None

    # 1. Split long text into chunks
    words = full_content.split()
    chunk_size = 1500 
    text_chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    logger.info(f"Content has been split into {len(text_chunks)} chunks for processing.")
    
    # 2. Map step: Process each chunk to generate a partial summary
    partial_summaries = []
    for i, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {i+1}/{len(text_chunks)}...")
        partial_summary = _summarize_chunk(chunk)
        if partial_summary:
            partial_summaries.append(f"Summary of Part {i+1}: {partial_summary}")
    
    if not partial_summaries:
        logger.error("Failed to generate partial summaries for any text chunk.")
        return None
        
    combined_partial_summaries = "\n\n".join(partial_summaries)
    logger.info("All partial summaries for the chunks have been generated.")

    # 3. Reduce step: Combine all partial summaries into a final, comprehensive summary
    logger.info("Generating final comprehensive summary (Reduce step)...")
    final_prompt = f"""
    You are a top-tier AI research analyst. I have divided a paper into sections and extracted a summary for each part.
    Your task is to synthesize these partial summaries into a single, high-quality, coherent technical summary of about 300-400 words.

    **Paper Title:** "{paper_title}"

    **Collected Partial Summaries:**
    ---
    {combined_partial_summaries}
    ---

    **Your Final Output Instructions:**
    Please write a professional summary that provides a comprehensive analysis of the paper, strictly following the structure below. Your output must clearly cover these four aspects:

    1.  **Background & Problem**: What problem is this paper trying to solve?
    2.  **Methodology**: What is the core method or model proposed by the authors?
    3.  **Key Contributions**: What are the most significant contributions of the paper?
    4.  **Experiments & Results**: What do the experimental results indicate?

    Begin writing your comprehensive summary now.
    """
    system_prompt = "You are a professional AI researcher and analyst."
    
    final_summary = llm_client_module.llm_client.generate(final_prompt, system_prompt)

    if not final_summary:
        logger.error(f"LLM failed to generate the final comprehensive summary for paper '{paper_title[:50]}...'.")
        return None

    logger.info(f"✅ --- Map-Reduce Summary Process Successfully Completed: Paper '{paper_title[:50]}...' ---")
    return final_summary.strip()