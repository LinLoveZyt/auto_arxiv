# hrag/embedding_engine.py

import logging
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


from core import config as config_module

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        logger.info("正在初始化嵌入引擎 (Embedding Engine)...")
        
        
        current_config = config_module.get_current_config()
        model_name = current_config['EMBEDDING_MODEL_NAME']
        device = current_config['EMBEDDING_DEVICE']

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"将使用设备: '{device}' 来加载嵌入模型。")
        
        try:
            logger.info(f"⏳ 正在从磁盘加载嵌入模型 '{model_name}'...")
            self.model = SentenceTransformer(model_name, device=device)
            self.max_seq_length = self.model.get_max_seq_length()
            logger.info(f"✅ 嵌入模型 '{model_name}' 加载成功。设备: {self.model.device}, 最大序列长度: {self.max_seq_length} tokens")
            self._initialized = True
        except Exception as e:
            logger.critical(f"❌ 无法加载嵌入模型 '{model_name}': {e}")
            self.__class__._instance = None
            raise RuntimeError("嵌入引擎初始化失败。")

    def encode(self, 
               texts: Union[str, List[str]], 
               prompt_name: Optional[str] = None,
               batch_size: int = 32) -> np.ndarray:
        """
        将一个或多个文本编码为嵌入向量，并自动进行批处理以节省显存。
        """
        if not self._initialized:
            raise RuntimeError("嵌入引擎未被成功初始化。")
        
        is_single_text = isinstance(texts, str)
        if is_single_text:
            texts = [texts]

        num_texts = len(texts)
        logger.info(f"开始对 {num_texts} 个文本进行编码 (内部批处理大小: {batch_size})...")

        try:
            embeddings = self.model.encode(
                sentences=texts,
                prompt_name=prompt_name,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            final_embeddings = embeddings[0] if is_single_text else embeddings
            logger.info(f"文本编码完成，生成了 {final_embeddings.shape} 维度的嵌入。")
            return final_embeddings
            
        except Exception as e:
            logger.error(f"文本编码过程中发生错误: {e}", exc_info=True)
            raise

embedding_engine: Optional[EmbeddingEngine] = None

def initialize_embedding_engine():
    """显式初始化嵌入引擎的函数。"""
    global embedding_engine
    if embedding_engine is None:
        try:
            embedding_engine = EmbeddingEngine()
        except RuntimeError:
            embedding_engine = None
    return embedding_engine
