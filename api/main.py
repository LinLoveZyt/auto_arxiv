# api/main.py


import logging
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List
from datetime import datetime, time
from pathlib import Path


from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from starlette.concurrency import iterate_in_threadpool

from core.bootstrap import initialize_core_services
from hrag.vector_db import vector_db_manager
from api import schemas
from workflows import daily_flow, query_flow
from data_ingestion import arxiv_fetcher
from hrag import metadata_db
from core import config as config_module
from agents import ingestion_agent
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes initialization on app startup and cleanup on shutdown."""
    logger.info("🚀 API service starting up...")
    # 初始化所有核心服务
    await run_in_threadpool(initialize_core_services)
    
    # ▼▼▼ [修改] ▼▼▼
    # 此处已移除服务器启动时自动运行每日任务的逻辑。
    # 现在每日任务只能通过API接口 /api/run/daily_workflow 手动触发。
    logger.info("Server started. Daily workflow will NOT run automatically.")
    
    yield
    
    logger.info("🛑 API service shutting down...")
    if vector_db_manager:
        await run_in_threadpool(vector_db_manager.save)
        logger.info("✅ FAISS index saved successfully.")

app = FastAPI(
    title="auto_arvix API",
    description="An intelligent system API for automated processing and querying of arXiv papers.",
    version="8.0.0 (Preference Edition)",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

app.mount("/ui", StaticFiles(directory="web"), name="ui")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", include_in_schema=False)
async def root(): 
    return RedirectResponse(url="/ui/index.html")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(): 
    return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title)

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint(): 
    return get_openapi(title=app.title, version=app.version, routes=app.routes)

# --- Workflow Triggers ---
@app.post("/api/run/daily_workflow", response_model=schemas.DailyRunResponse, tags=["Workflows"])
async def trigger_daily_run():
    result = await run_in_threadpool(daily_flow.run_daily_workflow)
    return schemas.DailyRunResponse(**result)

@app.post("/api/run/category_collection", response_model=schemas.GeneralStatusResponse, tags=["Workflows"])
async def trigger_category_collection():
    """
    手动触发一次轻量级的类别收集任务。
    该任务会从arXiv获取论文，仅进行AI分类以扩充系统知识，但不会下载PDF或入库。
    """
    logger.info("--- [API] 接收到手动类别收集请求 ---")
    # 注意：我们复用 schemas.GeneralStatusResponse 作为响应模型，因为它足够通用
    # 我们调用新创建的工作流函数
    result = await run_in_threadpool(daily_flow.run_category_collection_workflow)
    return schemas.GeneralStatusResponse(message=result.get("message", "操作完成。"))


@app.get("/api/settings/available-models", response_model=List[str], tags=["Configuration"])
async def get_available_models():
    """获取所有可用的LLM模型列表。"""
    return config_module.get_current_config().get("AVAILABLE_LLM_MODELS", [])


@app.post("/api/query", tags=["Core Functionality"])
async def handle_query_stream(request: schemas.QueryRequest):
    """
    Handles user's natural language query with a streaming response for real-time progress.
    """
    logger.info(f"Received streaming query request: '{request.query_text[:50]}...' (Online: {request.online_search_enabled})")

    workflow_instance = query_flow.QueryWorkflow()

    # 注意：这里不再传递 model_name
    sync_generator = workflow_instance.run_stream(
        query_text=request.query_text, 
        online_search_enabled=request.online_search_enabled
    )

    async_generator = iterate_in_threadpool(sync_generator)

    async def final_streamer():
        try:
            async for update in async_generator:
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            logger.critical(f"Unhandled exception during stream processing: {e}", exc_info=True)
            error_update = {"type": "error", "message": "An internal server error occurred while processing your request."}
            yield f"data: {json.dumps(error_update)}\n\n"

    return StreamingResponse(final_streamer(), media_type="text/event-stream")


@app.get("/api/reports", response_model=List[str], tags=["Reports"])
async def list_reports():
    """Lists all available daily PDF reports."""
    if not config_module.REPORTS_DIR.exists():
        return []
    try:
        # Sort files by name, which should roughly correspond to date
        files = sorted([f.name for f in config_module.REPORTS_DIR.iterdir() if f.suffix == '.pdf'], reverse=True)
        return files
    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        return []

        
@app.get("/api/reports/{filename}", tags=["Reports"])
async def get_report_pdf(filename: str):
    """Serves a specific report PDF file."""
    # Security: Ensure the filename is safe
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    report_path = config_module.REPORTS_DIR / filename
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
        
    return FileResponse(report_path, media_type='application/pdf', filename=filename)


# --- File Services ---
@app.get("/papers/pdf/{arxiv_id}", tags=["File Services"])
async def get_paper_pdf(arxiv_id: str):
    details = metadata_db.get_paper_details_by_id(arxiv_id)
    if not details or not details.get("pdf_path"): raise HTTPException(status_code=404, detail="PDF path not found")
    pdf_path = Path(details["pdf_path"])
    if not pdf_path.is_file(): raise HTTPException(status_code=404, detail="PDF file does not exist")
    return FileResponse(pdf_path, media_type='application/pdf', filename=f"{arxiv_id}.pdf")



@app.get("/api/categories", response_model=Dict[str, Any], tags=["Configuration"])
async def get_all_categories():
    """获取所有已知的领域和任务分类。"""
    if not config_module.CATEGORIES_JSON_PATH.exists():
        return {}
    try:
        with open(config_module.CATEGORIES_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"无法读取分类文件: {e}")
        raise HTTPException(status_code=500, detail="Could not read categories file.")

@app.get("/api/user/preferences", response_model=Dict[str, Any], tags=["Configuration"])
async def get_user_preferences():
    """获取用户当前已选择的偏好分类。"""
    if not config_module.USER_PREFERENCES_PATH.exists():
        return {"selected_categories": []}
    try:
        with open(config_module.USER_PREFERENCES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"无法读取用户偏好文件: {e}")
        raise HTTPException(status_code=500, detail="Could not read user preferences file.")

@app.post("/api/user/preferences", response_model=Dict[str, str], tags=["Configuration"])
async def save_user_preferences(payload: Dict[str, Any] = Body(...)):
    """保存用户选择的偏好分类。"""
    # ▼▼▼ [核心修改] 增加显式日志和更健壮的文件写入逻辑 ▼▼▼
    logger.info(f"--- [接收到偏好保存请求] --- 准备保存用户偏好...")
    
    # 确保目标目录存在，作为一道额外的保险
    try:
        config_module.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"在保存偏好前，无法创建或确认 storage 目录: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误：无法访问存储目录。")

    if "selected_categories" not in payload:
        logger.error(f"收到的偏好设置请求体格式错误，缺少 'selected_categories' 键。Payload: {payload}")
        raise HTTPException(status_code=400, detail="请求体格式错误。")

    try:
        with open(config_module.USER_PREFERENCES_PATH, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        
        num_selected = len(payload.get("selected_categories", []))
        logger.info(f"✅ --- [偏好保存成功] --- 用户偏好已写入 user_preferences.json。共保存 {num_selected} 条偏好。")
        return {"message": f"偏好保存成功！共保存 {num_selected} 条。"}
        
    except Exception as e:
        logger.critical(f"❌ 写入用户偏好文件时发生致命错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误：无法写入偏好文件。")


@app.post("/api/categories/propose-merges", response_model=Dict[str, Any], tags=["Maintenance"])
async def propose_category_merges():
    """
    触发AI分析现有分类，并返回一个合并建议列表。
    """
    logger.info("--- [API] 接收到分类合并建议请求 ---")
    try:
        proposals = await run_in_threadpool(ingestion_agent.propose_category_merges)
        if proposals is None:
            raise HTTPException(status_code=500, detail="AI Agent未能成功生成建议。")
        return proposals
    except Exception as e:
        logger.critical(f"生成合并建议时发生意外错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误。")


@app.get("/api/settings/global", response_model=Dict[str, Any], tags=["Configuration"])
async def get_global_settings():
    """获取当前所有可动态配置的全局参数。"""
    return config_module.get_current_config()

@app.post("/api/settings/global", response_model=Dict[str, str], tags=["Configuration"])
async def update_global_settings(payload: Dict[str, Any] = Body(...)):
    """更新全局参数，并将其写入覆盖文件。"""
    try:
        current_override = {}
        if config_module.CONFIG_OVERRIDE_PATH.exists():
            with open(config_module.CONFIG_OVERRIDE_PATH, 'r', encoding='utf-8') as f:
                current_override = json.load(f)
        
        current_override.update(payload)

        with open(config_module.CONFIG_OVERRIDE_PATH, 'w', encoding='utf-8') as f:
            json.dump(current_override, f, indent=4)
        
        logger.info(f"全局配置已成功更新，内容: {payload}")
        return {"message": "全局配置更新成功。新设置将在下一次任务运行时生效。"}
    except Exception as e:
        logger.error(f"更新全局配置文件时失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="写入配置文件时发生内部错误。")


@app.get("/api/settings/global", response_model=Dict[str, Any], tags=["Configuration"])
async def get_global_settings():
    """获取当前所有可动态配置的全局参数。"""
    return config_module.get_current_config()

@app.post("/api/settings/global", response_model=Dict[str, str], tags=["Configuration"])
async def update_global_settings(payload: Dict[str, Any] = Body(...)):
    """更新全局参数，并将其写入覆盖文件。"""
    try:
        # 读取现有的覆盖文件，如果不存在则创建一个空字典
        current_override = {}
        if config_module.CONFIG_OVERRIDE_PATH.exists():
            with open(config_module.CONFIG_OVERRIDE_PATH, 'r', encoding='utf-8') as f:
                current_override = json.load(f)
        
        # 使用传入的payload更新配置
        current_override.update(payload)

        # 将更新后的配置写回覆盖文件
        with open(config_module.CONFIG_OVERRIDE_PATH, 'w', encoding='utf-8') as f:
            json.dump(current_override, f, indent=4)
        
        logger.info(f"全局配置已成功更新，内容: {payload}")
        return {"message": "全局配置更新成功。新设置将在下一次任务运行时生效。"}
    except Exception as e:
        logger.error(f"更新全局配置文件时失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="写入配置文件时发生内部错误。")

@app.post("/api/categories/execute-merges", response_model=Dict[str, str], tags=["Maintenance"])
async def execute_category_merges(payload: Dict[str, Any] = Body(...)):
    """
    根据用户确认的列表，使用两阶段提交流程执行分类合并操作，以解决链式依赖问题。
    """
    logger.info("--- [API] 接收到执行分类合并请求 (v2 - 两阶段) ---")
    confirmed_merges = payload.get("confirmed_merges")
    if not isinstance(confirmed_merges, list):
        raise HTTPException(status_code=400, detail="请求体格式错误，需要'confirmed_merges'列表。")

    if not confirmed_merges:
        return {"message": "没有需要执行的合并操作。"}

    conn = metadata_db.get_db_connection()
    tasks_to_delete = set()
    successful_updates = 0
    try:
        # --- 阶段一：更新所有从属关系 (在单个事务中完成) ---
        logger.info("--- [合并阶段1/2] 开始更新所有分类的从属关系 ---")
        with conn:
            for merge_item in confirmed_merges:
                from_cat = merge_item.get("from")
                to_cat = merge_item.get("to")
                if not from_cat or not to_cat: continue

                from_id, to_id = metadata_db.execute_category_merge(
                    from_domain_name=from_cat["domain"], from_task_name=from_cat["task"],
                    to_domain_name=to_cat["domain"], to_task_name=to_cat["task"],
                    conn=conn
                )
                if from_id is not None and from_id != to_id:
                    tasks_to_delete.add(from_id)
                    successful_updates += 1
        logger.info("--- [合并阶段1/2] 所有从属关系更新完毕 ---")


        # --- 阶段二：删除所有冗余的Task (在另一个事务中完成) ---
        if tasks_to_delete:
            logger.info(f"--- [合并阶段2/2] 开始删除 {len(tasks_to_delete)} 个冗余任务 ---")
            with conn:
                for task_id in tasks_to_delete:
                    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            logger.info("--- [合并阶段2/2] 所有冗余任务删除完毕 ---")


        # --- 文件层面的清理和对齐 (在所有数据库操作成功后进行) ---
        all_categories = ingestion_agent.get_known_categories()
        merge_map = { (item["from"]["domain"], item["from"]["task"]): (item["to"]["domain"], item["to"]["task"]) for item in confirmed_merges }

        # a. 清理 categories.json
        cats_to_delete_tuples = { (item["from"]["domain"], item["from"]["task"]) for item in confirmed_merges }
        temp_categories = all_categories.copy()
        for domain, task in cats_to_delete_tuples:
            if domain in temp_categories and task in temp_categories[domain]["tasks"]:
                del temp_categories[domain]["tasks"][task]
                if not temp_categories[domain]["tasks"]:
                     del temp_categories[domain]
        # vvv [修改] vvv
        with open(config_module.CATEGORIES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(temp_categories, f, indent=4, ensure_ascii=False)
        logger.info("categories.json 文件已清理。")

        # b. 对齐 user_preferences.json
        if config_module.USER_PREFERENCES_PATH.exists():
            with open(config_module.USER_PREFERENCES_PATH, 'r', encoding='utf-8') as f:
                user_prefs = json.load(f)

            original_prefs = user_prefs.get("selected_categories", [])
            updated_prefs = [
                {"domain": merge_map.get((p["domain"], p["task"]), (p["domain"], p["task"]))[0],
                 "task": merge_map.get((p["domain"], p["task"]), (p["domain"], p["task"]))[1]}
                for p in original_prefs
            ]
            unique_prefs_tuples = sorted(list(set((d['domain'], d['task']) for d in updated_prefs)))
            user_prefs["selected_categories"] = [{"domain": d, "task": t} for d, t in unique_prefs_tuples]

            with open(config_module.USER_PREFERENCES_PATH, 'w', encoding='utf-8') as f:
                json.dump(user_prefs, f, indent=4, ensure_ascii=False)
            logger.info("user_preferences.json 文件已对齐。")
        # ^^^ [修改] ^^^

        message = f"分类清理完成，成功处理 {successful_updates} / {len(confirmed_merges)} 条合并规则。"
        logger.info(message)
        return {"message": message}

    except Exception as e:
        logger.critical(f"执行合并时发生意外错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误。")
    finally:
        if conn:
            conn.close()