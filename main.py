# main.py

import typer
import uvicorn
import logging
from typing import Optional

# ▼▼▼ [修改] 改变导入方式 ▼▼▼
from core import config as config_module
from core import logger as core_logger
from core.bootstrap import initialize_core_services
from api.main import app as fastapi_app
from workflows import daily_flow
from hrag import metadata_db

# ▼▼▼ [修改] 在脚本加载时获取一次配置，供Typer使用 ▼▼▼
cli_config = config_module.get_current_config()

# 创建一个Typer应用实例
cli = typer.Typer(help="auto_arvix 项目的命令行管理工具。")

# 在所有命令执行之前，先配置好日志
@cli.callback()
def main_callback():
    """
    初始化日志系统。这个函数会在任何命令执行之前被自动调用。
    """
    core_logger.setup_logging()


@cli.command(name="run-server", help="启动FastAPI服务器。")
def run_server(
    host: str = typer.Option(
        # ▼▼▼ [修改] 使用从字典中获取的值 ▼▼▼
        cli_config['API_HOST'], 
        "--host",
        help="服务器绑定的IP地址。"
    ),
    port: int = typer.Option(
        # ▼▼▼ [修改] 使用从字典中获取的值 ▼▼▼
        cli_config['API_PORT'], 
        "--port",
        help="服务器监听的端口。"
    ),
    reload: bool = typer.Option(
        False,
        "--reload/--no-reload",
        "-r/-R",
        help="开启/关闭热重载模式（开发时推荐开启）。"
    )
):
    """
    启动应用的主命令。
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"准备在 {host}:{port} 启动Uvicorn服务器...")
    typer.echo(f"🚀 FastAPI服务器即将启动。请在浏览器中访问 http://{host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

@cli.command(name="daily", help="独立地、手动执行一次每日任务，不启动服务器。")
def run_daily_only():
    """
    一个独立的命令，用于在不启动Web服务的情况下手动触发每日工作流。
    非常适合用于定时任务（如Linux下的cron job）。
    """
    logger = logging.getLogger(__name__)
    logger.info("通过命令行手动触发每日工作流...")
    
    try:
        initialize_core_services()
        logger.info("核心服务初始化完毕。")
    except Exception as e:
        logger.critical(f"为独立任务初始化核心服务时发生严重错误: {e}", exc_info=True)
        typer.echo("错误：核心服务初始化失败，请检查日志。")
        raise typer.Exit(code=1)

    try:
        daily_flow.run_daily_workflow()
        typer.echo("每日工作流执行完毕。")
    except Exception as e:
        logger.critical(f"手动执行每日工作流时发生严重错误: {e}", exc_info=True)
        typer.echo("错误：每日工作流执行失败，请检查日志。")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # 程序的主入口，将执行交给Typer
    cli()
