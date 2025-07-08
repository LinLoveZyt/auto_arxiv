# core/llm_client.py

import ollama
import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any


from core import config as config_module

logger = logging.getLogger(__name__)



class LLMClient:
    """
    A client that encapsulates interactions with the local Ollama large model.
    Uses a singleton pattern and an internal client with a timeout.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Initializing LLM client manager with timeout...")
        self.config = config_module.get_current_config()
        self.retry_attempts = self.config['LLM_JSON_RETRY_ATTEMPTS']
        
        # 实例化一个带超时的客户端
        self.client = ollama.Client(
            # host 参数可以根据需要配置，这里使用默认
            # host='http://127.0.0.1:11434', 
            timeout=self.config.get('OLLAMA_TIMEOUT', 120)
        )
        
        try:
            default_model = self.config['OLLAMA_MODEL_NAME']
            logger.info(f"Checking for default LLM model '{default_model}'...")
            # 使用带超时的客户端实例来检查模型
            self.client.show(default_model)
            logger.info(f"✅ Default LLM model '{default_model}' is available.")
            self._initialized = True
        except Exception as e:
            logger.error(f"❌ Default LLM model initialization failed: {e}")
            logger.error(f"Please ensure the Ollama service is running and the model '{self.config['OLLAMA_MODEL_NAME']}' has been downloaded.")
            self.__class__._instance = None
            raise ConnectionError(f"Cannot connect to Ollama model '{self.config['OLLAMA_MODEL_NAME']}'")

    def _parse_qwen3_output(self, text: str) -> tuple[str, str]:
        """从Qwen3的输出中分离思考过程和最终内容。"""
        think_match = re.search(r'<think>(.*?)<\/think>', text, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            content = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return thinking_content, content
        return "", text

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> Optional[str]:
        start_time = datetime.now()
        current_config = config_module.get_current_config()
        target_model = current_config['OLLAMA_MODEL_NAME']
        
        options = {'temperature': 0.7, 'num_ctx': 4096}
        if "qwen" in target_model.lower():
            options.update({'temperature': 0.6, 'top_p': 0.95})

        final_prompt = prompt
        if "qwen" in target_model.lower() and not current_config.get("ENABLE_THINKING_MODE", True):
            final_prompt += "\n/no_think"
            logger.info("'/no_think' command injected for Qwen model to disable thinking.")

        try:
            # 使用带超时的客户端实例
            response = self.client.chat(
                model=target_model,
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': final_prompt}],
                options=options
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            content = response['message']['content']
            tokens = response.get('eval_count', 0)
            tps = tokens / duration if duration > 0 else 0
            
            logger.info(f"LLM Response from '{target_model}': took {duration:.2f}s, generated {tokens} tokens ({tps:.2f} t/s).")

            if "qwen" in target_model.lower() and current_config.get("ENABLE_THINKING_MODE", True):
                thinking_content, final_content = self._parse_qwen3_output(content)
                if thinking_content:
                    logger.info(f"[Qwen3 Thinking]: {thinking_content[:500]}...")
                return final_content
                
            return content

        except Exception as e:
            logger.error(f"Error during LLM call with model {target_model}: {e}")
            return None

    def generate_json(self, prompt: str, system_prompt: str = "You are a helpful JSON assistant.") -> Optional[Dict[str, Any]]:
        for attempt in range(self.retry_attempts):
            current_config = config_module.get_current_config()
            target_model = current_config['OLLAMA_MODEL_NAME']

            logger.info(f"Attempting to generate JSON with model '{target_model}' (attempt {attempt + 1}/{self.retry_attempts})...")
            
            final_prompt = prompt
            if "qwen" in target_model.lower():
                final_prompt += "\n/no_think"
                logger.info("'/no_think' command forcefully injected for JSON generation task.")
            
            try:
                 # 使用带超时的客户端实例
                 response = self.client.chat(
                    model=target_model,
                    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': final_prompt}],
                    options={'temperature': 0.1, 'top_p': 0.8}
                )
                 raw_response = response['message']['content']
            except Exception as e:
                logger.error(f"LLM call failed during JSON generation: {e}")
                raw_response = None

            if not raw_response:
                logger.warning("LLM returned an empty response, retrying...")
                continue

            try:
                match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_response)
                if match:
                    json_part = match.group(1).strip()
                else:
                    json_part = self._extract_json_from_string(raw_response)

                if not json_part:
                    raise ValueError("Could not extract valid JSON content from LLM response.")
                
                parsed_json = json.loads(json_part)
                logger.info("Successfully parsed JSON.")
                return parsed_json
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON: {e}. LLM raw response:\n---\n{raw_response}\n---")
                if attempt < self.retry_attempts - 1:
                    logger.warning("Retrying...")
                else:
                    logger.error("Max retry attempts reached, failed to get valid JSON.")
                    return None
        return None

    def _extract_json_from_string(self, text: str) -> str:
        """A helper function to extract a JSON part from an irregular string."""
        first_bracket_pos = -1
        for i, char in enumerate(text):
            if char in "{[":
                first_bracket_pos = i
                break

        if first_bracket_pos == -1: return ""

        last_bracket_pos = -1
        for i, char in enumerate(reversed(text)):
            if char in "}]":
                last_bracket_pos = len(text) - 1 - i
                break
        
        if last_bracket_pos == -1 or last_bracket_pos < first_bracket_pos: return ""
            
        return text[first_bracket_pos : last_bracket_pos + 1].strip()


llm_client: Optional[LLMClient] = None

def initialize_llm_client():
    """Function to explicitly initialize the LLM client."""
    global llm_client
    if llm_client is None:
        try:
            llm_client = LLMClient()
        except ConnectionError:
            llm_client = None
    return llm_client
