# core/config.py

import os
import json
from pathlib import Path

# --- 项目根目录 ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 1. 路径配置 (通常不动态修改) ---
STORAGE_DIR = BASE_DIR / "storage"
PAPER_PDF_DIR = STORAGE_DIR / "papers"
STRUCTURED_DATA_DIR = STORAGE_DIR / "structured_data"
DATABASE_DIR = STORAGE_DIR / "database"
REPORTS_DIR = STORAGE_DIR / "reports"
LOGS_DIR = STORAGE_DIR / "logs"
CONFIG_OVERRIDE_PATH = BASE_DIR / "config_override.json"
MONKEY_OCR_PATH = BASE_DIR.parent / "MonkeyOCR"
METADATA_DB_PATH = DATABASE_DIR / "metadata.sqlite3"
FAISS_INDEX_PATH = DATABASE_DIR / "hrag_index.faiss"
CATEGORIES_JSON_PATH = STORAGE_DIR / "categories.json"
USER_PREFERENCES_PATH = STORAGE_DIR / "user_preferences.json"
DAILY_REPORT_PREFIX = "Daily_arXiv_Report"


# --- 2. 默认配置字典 ---
# 我们将所有可动态配置的参数放入一个字典中，方便管理和覆盖
DEFAULT_CONFIG = {
    # LLM and Embedding
    "AVAILABLE_LLM_MODELS": ["auto-arvix-unsloth-pro", "auto-arxiv-qwen3-16k"],
    "OLLAMA_MODEL_NAME": "auto-arvix-unsloth-pro",
    "OLLAMA_TIMEOUT": 300, # 全局LLM请求超时（秒）
    "LLM_JSON_RETRY_ATTEMPTS": 3,
    "ENABLE_THINKING_MODE": True,

    "EMBEDDING_MODEL_NAME": "Qwen3-Embedding-0.6B",
    "EMBEDDING_DEVICE": "auto",

    # Reranker
    "RERANKER_MODEL_NAME": "Qwen/Qwen3-Reranker-0.6B",
    
    # Agent Settings
    "MAX_TASKS_FOR_PROMPT": 50, # 分类合并建议时，发送给LLM的最大任务数
    "EMBEDDING_BATCH_SIZE": 64, 

    # Data Fetching & Processing
    "DAILY_PAPER_PROCESS_LIMIT": 20,
    "USER_QUERY_FETCH_LIMIT": 1,
    "ONLINE_SEARCH_PAPER_LIMIT": 10,
    "DEFAULT_ARXIV_DOMAINS": ["cs.AI", "cs.CV"],
    "PDF_PARSING_STRATEGY": "monkey", 
    
    # H-RAG
    "TOP_K_RESULTS": 15,
    "MAX_RELEVANT_PAPERS": 5,
    
    # Cold Start
    "COLD_START_PAPER_COUNT": 10,
    "COLD_START_DOMAINS": ["cs.AI", "cs.CV"],
    "COLD_START_YEARS_WINDOW": 5,
    
    # Report
    "REPORT_AUTHOR": "Auto-ARVIX Project",
    
    # API Server
    "API_HOST": "0.0.0.0",
    "API_PORT": 5002,
}

# --- 3. 动态配置加载函数 ---
def get_current_config() -> dict:
    """
    获取当前的有效配置。
    它首先加载默认配置，然后尝试用 config_override.json 的内容进行覆盖。
    这是整个动态配置机制的核心。
    """
    config = DEFAULT_CONFIG.copy()
    if CONFIG_OVERRIDE_PATH.exists():
        try:
            with open(CONFIG_OVERRIDE_PATH, 'r', encoding='utf-8') as f:
                override_config = json.load(f)
                config.update(override_config)
        except (json.JSONDecodeError, IOError):
            # 文件存在但格式错误或无法读取，忽略并使用默认值
            pass
    return config

# --- 4. 确保目录存在 ---
def create_directories():
    """Ensures all necessary storage directories exist."""
    dirs = [STORAGE_DIR, PAPER_PDF_DIR, STRUCTURED_DATA_DIR, DATABASE_DIR, REPORTS_DIR, LOGS_DIR]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

create_directories()
