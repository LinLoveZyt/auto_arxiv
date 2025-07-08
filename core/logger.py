# core/logger.py

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from core import config

# --- 日志格式 ---
# 定义了日志消息的结构：时间 - 日志级别 - 日志记录器名称 - 消息
LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
# 定义了彩色日志的颜色映射
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

def setup_logging():
    """
    配置全局日志记录器。
    - 向控制台输出带颜色的日志。
    - 向文件输出日志，并按天轮换。
    """
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # 设置根日志级别为INFO

    # --- 1. 控制台处理器 (StreamHandler) ---
    # 尝试使用 colorlog 创建彩色的控制台输出，如果失败则回退到标准输出
    try:
        import colorlog
        # 创建一个彩色的格式化器
        console_formatter = colorlog.ColoredFormatter(
            f"%(log_color)s{LOG_FORMAT}",
            log_colors=LOG_COLORS,
            reset=True,
            style='%'
        )
        print("彩色日志模块 (colorlog) 加载成功。")
    except ImportError:
        console_formatter = logging.Formatter(LOG_FORMAT)
        print("彩色日志模块 (colorlog) 未安装，将使用标准日志格式。")

    # 创建一个流处理器，将日志输出到标准错误流 (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # --- 2. 文件处理器 (FileHandler) ---
    # 确保日志目录存在
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = config.LOGS_DIR / "app.log"

    # 创建一个文件处理器，将日志写入文件
    # TimedRotatingFileHandler 可以让日志文件按时间自动分割
    # when='midnight' 表示每天午夜进行一次轮换
    # backupCount=7 表示保留最近7天的日志文件
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)

    # --- 将处理器添加到根日志记录器 ---
    # 添加前先清空已有的处理器，防止重复添加
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info("日志系统初始化完成，将同时输出到控制台和文件。")
    logging.info(f"日志文件路径: {log_file_path}")

