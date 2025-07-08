# hrag/reranker.py

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional

from core import config as config_module

logger = logging.getLogger(__name__)

class Reranker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        current_config = config_module.get_current_config()
        model_name = current_config['RERANKER_MODEL_NAME']
        
        logger.info(f"Initializing Reranker with model '{model_name}'...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(self.device).eval()

            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.max_length = 8192

            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            self._initialized = True
            logger.info(f"✅ Reranker initialized successfully on device '{self.device}'.")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Reranker: {e}", exc_info=True)
            self.__class__._instance = None
            raise RuntimeError("Reranker initialization failed.")

    def _format_instruction(self, query, doc, instruction):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        if not self._initialized:
            raise RuntimeError("Reranker is not initialized.")
        
        pairs = [self._format_instruction(query, doc, instruction) for doc in documents]
        inputs = self._process_inputs(pairs)
        
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        stacked_scores = torch.stack([false_vector, true_vector], dim=1)
        log_softmax_scores = torch.nn.functional.log_softmax(stacked_scores, dim=1)
        scores = log_softmax_scores[:, 1].exp().tolist()
        
        return scores

reranker: Optional[Reranker] = None

def initialize_reranker():
    global reranker
    if reranker is None:
        try:
            reranker = Reranker()
        except RuntimeError:
            reranker = None
    return reranker