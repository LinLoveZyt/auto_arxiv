# data_ingestion/pdf_processor.py

import logging
import json
import httpx
import time
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# ▼▼▼ [修改] 新增 unstructured 的导入 ▼▼▼
from unstructured.partition.pdf import partition_pdf

from core import config

logger = logging.getLogger(__name__)


def download_pdf(pdf_url: str, arxiv_id: str) -> Optional[Path]:
    """健壮的PDF下载函数，包含超时和进度反馈。"""
    
    safe_arxiv_id = arxiv_id.replace('/', '_')
    pdf_path = config.PAPER_PDF_DIR / f"{safe_arxiv_id}.pdf"
    
    if pdf_path.exists():
        logger.info(f"PDF file '{pdf_path.name}' already exists, skipping download.")
        return pdf_path
    
    logger.info(f"Downloading PDF from {pdf_url}...")
    try:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        with httpx.stream("GET", pdf_url, follow_redirects=True, timeout=30.0) as response:
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"✅ PDF file successfully saved to: {pdf_path}")
        return pdf_path
    except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
        logger.error(f"❌ Error downloading PDF: {e}")
        if pdf_path.exists(): pdf_path.unlink(missing_ok=True)
        return None


def _parse_with_monkey_ocr(pdf_path: Path, arxiv_id: str) -> Optional[Path]:
    """
    Parses a PDF using the external MonkeyOCR tool.
    (此函数保持不变)
    """
    safe_arxiv_id = arxiv_id.replace('/', '_')
    
    monkey_input_dir = config.MONKEY_OCR_PATH / "input" / "paper_pdf"
    monkey_output_dir = config.MONKEY_OCR_PATH / "output" / safe_arxiv_id
    monkey_input_pdf_path = monkey_input_dir / pdf_path.name
    
    final_json_path = config.STRUCTURED_DATA_DIR / f"{safe_arxiv_id}.json"
    final_images_dir = config.STRUCTURED_DATA_DIR / safe_arxiv_id / "images"

    if final_json_path.exists():
        logger.info(f"Final structured JSON '{final_json_path.name}' already exists, skipping MonkeyOCR parsing.")
        return final_json_path

    logger.info(f"--- Starting PDF processing with MonkeyOCR for {arxiv_id} ---")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Emptied CUDA cache before calling MonkeyOCR to maximize available memory.")

        monkey_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_path, monkey_input_pdf_path)
        logger.info(f"Copied PDF to MonkeyOCR input: {monkey_input_pdf_path}")

        logger.info("Executing MonkeyOCR script...")
        relative_input_path = Path("input") / "paper_pdf" / pdf_path.name
        
        command = [
            "conda", "run", "-n", "MonkeyOCR",
            "python", "parse.py", str(relative_input_path)
        ]
        
        process = subprocess.run(
            command,
            cwd=config.MONKEY_OCR_PATH,
            capture_output=True,
            text=True,
            check=False
        )

        if process.returncode != 0:
            logger.error(f"MonkeyOCR script failed for {arxiv_id} with return code {process.returncode}.")
            logger.error(f"MonkeyOCR STDOUT:\n{process.stdout}")
            logger.error(f"MonkeyOCR STDERR:\n{process.stderr}")
            return None
        
        logger.info(f"MonkeyOCR script executed successfully for {arxiv_id}.")

        expected_json = monkey_output_dir / f"{safe_arxiv_id}_content_list.json"
        expected_images_dir = monkey_output_dir / "images"

        if not expected_json.exists():
            logger.error(f"MonkeyOCR output JSON not found at {expected_json}")
            return None

        final_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(expected_json, final_json_path)
        logger.info(f"Copied result JSON to: {final_json_path}")

        if expected_images_dir.exists():
            if final_images_dir.exists():
                shutil.rmtree(final_images_dir)
            shutil.copytree(expected_images_dir, final_images_dir)
            logger.info(f"Copied images directory to: {final_images_dir}")

        return final_json_path

    except Exception as e:
        logger.critical(f"An unexpected error occurred during MonkeyOCR processing for {arxiv_id}: {e}", exc_info=True)
        return None
    finally:
        logger.info(f"Cleaning up temporary files for {arxiv_id} in MonkeyOCR directory...")
        if monkey_input_pdf_path.exists():
            monkey_input_pdf_path.unlink()
        if monkey_output_dir.exists():
            shutil.rmtree(monkey_output_dir)
        logger.info(f"--- Finished PDF processing with MonkeyOCR for {arxiv_id} ---")


def _parse_with_unstructured(pdf_path: Path, arxiv_id: str) -> Optional[Path]:
    """
    使用 unstructured 库和 'fast' 策略来解析PDF。
    它不提取图片，主要关注快速提取文本内容。
    """
    safe_arxiv_id = arxiv_id.replace('/', '_')
    json_path = config.STRUCTURED_DATA_DIR / f"{safe_arxiv_id}.json"

    # 注意：这里我们假设MonkeyOCR和unstructured的输出文件名相同。
    # 如果一个策略已经生成了文件，另一个策略将不会重复执行。
    if json_path.exists():
        logger.info(f"Structured JSON file '{json_path.name}' already exists, skipping 'unstructured' parsing.")
        return json_path

    logger.info(f"--- Starting PDF processing with unstructured (fast) for {arxiv_id} ---")
    try:
        # 使用unstructured进行解析
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",  # 明确使用快速策略
            infer_table_structure=True, # 尝试推断表格结构
            extract_images_in_pdf=False # 快速模式下不提取图片以提高速度
        )
        
        # 将解析出的元素转换为与MonkeyOCR兼容的字典列表格式
        # 我们只提取文本内容，并赋予一个 'text' 类型
        structured_data = []
        for el in elements:
            # 将unstructured的元素转换为我们项目中统一的、更简单的格式
            # 这样下游的摘要等agent就无需关心数据源是MonkeyOCR还是unstructured
            chunk = {
                "type": "text", # 统一类型为 'text'
                "text": el.text,
                # 可以选择性地保留一些元数据
                "metadata": {
                    "source": "unstructured",
                    "category": el.category,
                    "page_number": getattr(el.metadata, 'page_number', None)
                }
            }
            structured_data.append(chunk)

        # 保存为JSON文件
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"✅ PDF successfully parsed with unstructured (fast) and saved to: {json_path}")
        return json_path

    except Exception as e:
        logger.error(f"❌ Error during 'unstructured' (fast) parsing for {arxiv_id}: {e}", exc_info=True)
        if json_path.exists():
            json_path.unlink(missing_ok=True)
        return None


def process_paper(paper_data: Dict[str, Any], strategy: str) -> Optional[Dict[str, Path]]:
    """
    处理单篇论文：下载PDF并根据指定策略进行解析。

    Args:
        paper_data (Dict[str, Any]): 包含论文元数据的字典。
        strategy (str): 要使用的解析策略 ('monkey' 或 'fast')。

    Returns:
        Optional[Dict[str, Path]]: 包含PDF和JSON路径的字典，如果失败则返回None。
    """
    arxiv_id, pdf_url = paper_data.get("arxiv_id"), paper_data.get("pdf_url")
    if not arxiv_id or not pdf_url:
        logger.error(f"Paper data is incomplete: {paper_data.get('title')}")
        return None

    logger.info(f"--- Starting full processing pipeline for paper: {arxiv_id} ---")
    
    # 1. 下载PDF
    local_pdf_path = download_pdf(pdf_url, arxiv_id)
    if not local_pdf_path:
        logger.error(f"Processing failed: Could not download PDF for {arxiv_id}.")
        return None

    json_path = None
    # 2. 根据策略选择解析器
    if strategy == 'monkey':
        logger.info(f"Using 'monkey' strategy to parse PDF: {local_pdf_path.name}")
        json_path = _parse_with_monkey_ocr(local_pdf_path, arxiv_id)
    elif strategy == 'fast':
        logger.info(f"Using 'fast' strategy to parse PDF: {local_pdf_path.name}")
        json_path = _parse_with_unstructured(local_pdf_path, arxiv_id)
    else:
        logger.error(f"Unknown PDF parsing strategy: '{strategy}'")
        return None

    # 3. 检查解析结果
    if not json_path:
        logger.error(f"Processing failed: Could not parse PDF for {arxiv_id} using strategy '{strategy}'.")
        return None
        
    logger.info(f"--- Paper {arxiv_id} processed successfully ---")
    return {"pdf_path": local_pdf_path, "json_path": json_path}