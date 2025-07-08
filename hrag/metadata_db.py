# hrag/metadata_db.py

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from core import config

logger = logging.getLogger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """
    建立并返回一个数据库连接。
    - 开启外键约束支持。
    - 设置行工厂，使得查询结果可以像字典一样通过列名访问。
    """
    try:
        conn = sqlite3.connect(
            config.METADATA_DB_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as e:
        logger.critical(f"❌ 无法连接到SQLite数据库: {e}")
        raise

def create_tables():
    """
    创建所有必要的数据库表。
    如果表已存在，则不会执行任何操作。
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    representative_papers TEXT,
                    domain_id INTEGER NOT NULL,
                    FOREIGN KEY (domain_id) REFERENCES domains (id) ON DELETE CASCADE,
                    UNIQUE (name, domain_id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    authors TEXT,
                    summary TEXT,
                    generated_summary TEXT,
                    published_date TEXT,
                    added_date TIMESTAMP NOT NULL,
                    pdf_path TEXT,
                    structured_text_path TEXT,
                    domain_id INTEGER,
                    task_id INTEGER,
                    FOREIGN KEY (domain_id) REFERENCES domains (id) ON DELETE SET NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE SET NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    id INTEGER PRIMARY KEY,
                    type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    chunk_seq INTEGER,
                    domain_id INTEGER,
                    task_id INTEGER,
                    content_preview TEXT
                );
            """)
        logger.info("✅ 数据库表创建/验证成功。")
    except sqlite3.Error as e:
        logger.error(f"❌ 创建数据库表时出错: {e}")
    finally:
        conn.close()

def add_paper(paper_data: Dict[str, Any]) -> Optional[int]:
    """将一篇新论文的元数据添加到数据库。"""
    conn = get_db_connection()

    published_date_obj = paper_data.get("published_date")
    published_date_str = published_date_obj.isoformat() if isinstance(published_date_obj, datetime) else str(published_date_obj)

    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO papers (arxiv_id, title, authors, summary, published_date, added_date, pdf_path, structured_text_path)
                VALUES (:arxiv_id, :title, :authors, :summary, :published_date, :added_date, :pdf_path, :structured_text_path)
                """,
                {
                    "arxiv_id": paper_data["arxiv_id"],
                    "title": paper_data["title"],
                    "authors": json.dumps(paper_data.get("authors", [])),
                    "summary": paper_data.get("summary"),
                    "published_date": published_date_str,
                    "added_date": datetime.now(),
                    "pdf_path": str(paper_data.get("pdf_path")),
                    "structured_text_path": str(paper_data.get("json_path")),
                }
            )
            logger.info(f"成功添加论文到数据库: {paper_data['arxiv_id']}")
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        logger.warning(f"论文 {paper_data['arxiv_id']} 已存在于数据库中，跳过添加。")
        return None
    except sqlite3.Error as e:
        logger.error(f"添加论文 {paper_data['arxiv_id']} 到数据库时出错: {e}")
        return None
    finally:
        conn.close()

def _manage_connection(func):
    """一个装饰器，用于简化函数内部的连接管理。"""
    def wrapper(*args, **kwargs):
        conn = kwargs.get('conn')
        is_external_conn = conn is not None

        if not is_external_conn:
            kwargs['conn'] = get_db_connection()

        try:
            return func(*args, **kwargs)
        finally:
            if not is_external_conn and kwargs.get('conn'):
                kwargs['conn'].close()
    return wrapper

@_manage_connection
def add_or_get_domain(name: str, description: str = "", *, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    """添加一个新领域，或获取已存在领域的ID。可在事务中运行。"""
    try:
        cursor = conn.execute("SELECT id FROM domains WHERE name = ?", (name,))
        domain = cursor.fetchone()
        if domain:
            return domain["id"]

        cursor = conn.execute("INSERT INTO domains (name, description) VALUES (?, ?)", (name, description))
        logger.info(f"创建了新领域: {name}")
        return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"处理领域 '{name}' 时出错: {e}")
        return None

@_manage_connection
def add_or_get_task(name: str, domain_id: int, description: str = "", *, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    """在特定领域下添加一个新任务，或获取已存在任务的ID。可在事务中运行。"""
    try:
        cursor = conn.execute("SELECT id FROM tasks WHERE name = ? AND domain_id = ?", (name, domain_id))
        task = cursor.fetchone()
        if task:
            return task["id"]

        cursor = conn.execute(
            "INSERT INTO tasks (name, domain_id, description, representative_papers) VALUES (?, ?, ?, ?)",
            (name, domain_id, description, json.dumps({}))
        )
        logger.info(f"在领域ID {domain_id} 下创建了新任务: {name}")
        return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"处理任务 '{name}' 时出错: {e}")
        return None

@_manage_connection
def add_vector_metadata_batch(metadata_list: List[Dict[str, Any]], *, conn: Optional[sqlite3.Connection] = None):
    """批量添加向量元数据。可在事务中运行。"""
    try:
        conn.executemany(
            """
            INSERT INTO vector_metadata (id, type, source_id, chunk_seq, domain_id, task_id, content_preview)
            VALUES (:id, :type, :source_id, :chunk_seq, :domain_id, :task_id, :content_preview)
            """,
            metadata_list
        )
        logger.info(f"成功批量添加 {len(metadata_list)} 条向量元数据。")
    except sqlite3.Error as e:
        logger.error(f"批量添加向量元数据时出错: {e}")
        raise

@_manage_connection
def update_paper_summary_and_classification(arxiv_id: str, domain_id: int, task_id: int, summary: str, *, conn: Optional[sqlite3.Connection] = None):
    """更新一篇已存在论文的分类信息和AI生成的摘要。可在事务中运行。"""
    try:
        conn.execute(
            "UPDATE papers SET domain_id = ?, task_id = ?, generated_summary = ? WHERE arxiv_id = ?",
            (domain_id, task_id, summary, arxiv_id)
        )
        logger.info(f"成功更新论文 {arxiv_id} 的分类和摘要信息。")
    except sqlite3.Error as e:
        logger.error(f"更新论文 {arxiv_id} 信息时出错: {e}")
        raise

def check_if_paper_exists(arxiv_id: str) -> bool:
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        return cursor.fetchone() is not None
    finally:
        conn.close()

def get_all_domains_and_tasks() -> Dict[str, List[str]]:
    conn = get_db_connection()
    try:
        query = """
            SELECT d.name as domain_name, t.name as task_name
            FROM domains d
            LEFT JOIN tasks t ON d.id = t.domain_id
            ORDER BY d.name, t.name;
        """
        cursor = conn.execute(query)
        result = {}
        for row in cursor.fetchall():
            if row["domain_name"] not in result:
                result[row["domain_name"]] = []
            if row["task_name"]:
                result[row["domain_name"]].append(row["task_name"])
        return result
    finally:
        conn.close()

def get_metadata_for_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids: return {}
    conn = get_db_connection()
    try:
        placeholders = ','.join('?' for _ in ids)
        query = f"SELECT * FROM vector_metadata WHERE id IN ({placeholders})"
        cursor = conn.execute(query, ids)
        return {row['id']: dict(row) for row in cursor.fetchall()}
    finally:
        conn.close()

def get_paper_details_by_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        row = cursor.fetchone()
        if row:
            paper_dict = dict(row)
            paper_dict['authors'] = json.loads(paper_dict.get('authors', '[]'))
            return paper_dict
        return None
    finally:
        conn.close()

def check_if_today_papers_exist() -> bool:
    conn = get_db_connection()
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cursor = conn.execute(
            "SELECT 1 FROM papers WHERE added_date >= ? LIMIT 1",
            (today_start,)
        )
        exists = cursor.fetchone() is not None
        logger.info(f"检查今日是否已处理论文: {'是' if exists else '否'}")
        return exists
    finally:
        conn.close()


@_manage_connection
def get_max_vector_id(*, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    """从元数据表中获取当前最大的向量ID。"""
    try:
        cursor = conn.execute("SELECT MAX(id) FROM vector_metadata")
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else -1
    except sqlite3.Error as e:
        logger.error(f"获取最大向量ID时出错: {e}")
        return None

def get_total_paper_count() -> int:
    """获取数据库中论文的总数。"""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT COUNT(id) FROM papers")
        result = cursor.fetchone()
        return result[0] if result else 0
    except sqlite3.Error as e:
        logger.error(f"获取论文总数时出错: {e}")
        return 0
    finally:
        conn.close()

@_manage_connection
def execute_category_merge(
    from_domain_name: str, from_task_name: str,
    to_domain_name: str, to_task_name: str,
    *, conn: Optional[sqlite3.Connection] = None
) -> tuple[Optional[int], Optional[int]]:
    """
    在数据库中更新一个分类的从属关系，但不删除。
    返回 from_task_id 和 to_task_id。
    """
    logger.info(f"正在重新映射分类: ('{from_domain_name}', '{from_task_name}') -> ('{to_domain_name}', '{to_task_name}')")
    try:
        # 1. 确保 'to' 分类存在，并获取其ID
        to_domain_id = add_or_get_domain(to_domain_name, conn=conn)
        to_task_id = add_or_get_task(to_task_name, to_domain_id, conn=conn)

        # 2. 获取 'from' 分类的ID
        cursor = conn.execute("SELECT id FROM domains WHERE name = ?", (from_domain_name,))
        from_domain_row = cursor.fetchone()
        if not from_domain_row:
             logger.error(f"无法找到源领域 '{from_domain_name}'。")
             return None, None
        from_domain_id = from_domain_row['id']

        cursor = conn.execute("SELECT id FROM tasks WHERE name = ? AND domain_id = ?", (from_task_name, from_domain_id))
        from_task_row = cursor.fetchone()
        if not from_task_row:
            logger.error(f"无法找到源任务 '{from_task_name}' 在领域 '{from_domain_name}' 中。")
            return None, None
        from_task_id = from_task_row['id']

        if from_task_id == to_task_id:
             logger.warning(f"源分类和目标分类ID相同 ({from_task_id})，无需合并。")
             return from_task_id, to_task_id

        # 3. 更新 papers 和 vector_metadata 表
        conn.execute("UPDATE papers SET domain_id = ?, task_id = ? WHERE task_id = ?",
                     (to_domain_id, to_task_id, from_task_id))
        conn.execute("UPDATE vector_metadata SET domain_id = ?, task_id = ? WHERE task_id = ?",
                     (to_domain_id, to_task_id, from_task_id))

        logger.info(f"已将任务ID {from_task_id} 的所有条目重新指向任务ID {to_task_id}。")
        return from_task_id, to_task_id

    except sqlite3.Error as e:
        logger.error(f"执行分类数据库合并时发生错误: {e}", exc_info=True)
        return None, None

create_tables()