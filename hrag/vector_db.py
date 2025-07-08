# hrag/vector_db.py

import logging
import faiss
import numpy as np
import os
import threading
from typing import Optional

from core import config as config_module
# --- 核心修改：导入整个模块，而不是模块内的变量 ---
from hrag import embedding_engine as embedding_engine_module

logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    向量数据库管理器，封装了与FAISS的所有交互。
    - 自动检测并使用GPU（如果可用）。
    - 支持从磁盘加载/保存索引。
    - 采用线程安全的单例模式。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VectorDBManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return

            logger.info("正在初始化向量数据库管理器 (Vector DB Manager)...")
            
            # --- 核心修改：通过模块访问实例，确保获取最新值 ---
            if not embedding_engine_module.embedding_engine:
                raise RuntimeError("Embedding Engine未能初始化，VectorDB无法确定向量维度。请检查启动顺序。")

            self.index = None
            self.is_gpu_index = False
            self.dimension = embedding_engine_module.embedding_engine.model.get_sentence_embedding_dimension()
            
            self._load_or_create_index()
            self._initialized = True

    def _load_or_create_index(self):
        """加载或创建FAISS索引，并根据环境自动选择CPU/GPU。"""
        if os.path.exists(config_module.FAISS_INDEX_PATH):
            logger.info(f"正在从 '{config_module.FAISS_INDEX_PATH}' 加载已存在的FAISS索引...")
            try:
                self.index = faiss.read_index(str(config_module.FAISS_INDEX_PATH))
                if self.index.d != self.dimension:
                    raise ValueError(f"索引维度({self.index.d})与模型维度({self.dimension})不匹配!")
                logger.info(f"✅ 成功加载索引，当前包含 {self.index.ntotal} 个向量。")
            except Exception as e:
                logger.error(f"❌ 加载FAISS索引失败: {e}。将创建一个新索引。")
                self._create_new_index()
        else:
            self._create_new_index()
        
        if faiss.get_num_gpus() > 0:
            try:
                self._move_to_gpu()
            except Exception as e:
                logger.error(f"❌ 将索引移动到GPU时失败: {e}。将继续使用CPU。")

    def _create_new_index(self):
        """创建一个新的CPU FAISS索引。"""
        logger.info(f"未找到现有索引，正在创建一个新的FAISS索引，维度为 {self.dimension}。")
        # 使用IndexFlatL2，它适用于大多数场景，是其他更复杂索引的基础
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def _move_to_gpu(self):
        """将CPU索引转换为GPU索引以加速搜索。"""
        logger.info(f"检测到 {faiss.get_num_gpus()} 个可用GPU，正在将索引转移到GPU...")
        # 创建一个标准的GPU资源对象
        res = faiss.StandardGpuResources()
        # 将CPU索引转换为GPU索引
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index) # 使用第一个GPU (ID 0)
        self.is_gpu_index = True
        logger.info("✅ 索引已成功转移到GPU。")

    def add(self, vectors: np.ndarray):
        """向索引中添加新的向量。"""
        if not self.index:
            logger.error("索引未初始化，无法添加向量。")
            return
            
        # FAISS需要float32类型的numpy数组
        vectors = vectors.astype('float32')
        with self._lock:
            self.index.add(vectors)
        logger.info(f"向索引中添加了 {len(vectors)} 个新向量。当前总数: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        在索引中搜索与查询向量最相似的k个向量。

        Args:
            query_vector (np.ndarray): 查询向量，形状应为 (1, dimension)。
            k (int): 希望返回的最相似邻居的数量。

        Returns:
            tuple[np.ndarray, np.ndarray]: (距离, 索引ID) 的元组。
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("索引为空或未初始化，无法执行搜索。")
            return np.array([]), np.array([])
        
        query_vector = query_vector.astype('float32')
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        with self._lock:
            distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def save(self):
        """
        将索引安全地保存到磁盘。
        如果索引在GPU上，会先转换回CPU再保存。
        """
        with self._lock:
            if not self.index:
                logger.error("索引未初始化，无法保存。")
                return

            logger.info(f"正在将索引保存到 '{config_module.FAISS_INDEX_PATH}'...")
            
            if self.is_gpu_index:
                logger.info("索引在GPU上，正在转换回CPU以便保存...")
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            
            faiss.write_index(cpu_index, str(config_module.FAISS_INDEX_PATH))
            logger.info(f"✅ 索引已成功保存，包含 {cpu_index.ntotal} 个向量。")

# 全局变量和初始化函数保持不变
vector_db_manager: Optional[VectorDBManager] = None

def initialize_vector_db():
    """显式初始化向量数据库管理器的函数。"""
    global vector_db_manager
    if vector_db_manager is None:
        try:
            vector_db_manager = VectorDBManager()
        except (RuntimeError, Exception) as e:
            logger.critical(f"无法创建向量数据库管理器单例: {e}")
            vector_db_manager = None
    return vector_db_manager