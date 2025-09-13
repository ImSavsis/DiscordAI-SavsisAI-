import requests
import json
import asyncio
from typing import Optional, Dict, Any, List
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

class LMStudioClient:
    def __init__(self, base_url: str = "http://10.33.0.2:1234", model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, 
                              prompt: str, 
                              max_tokens: int = 500,
                              temperature: float = 0.7,
                              system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                logger.info(f"LM Studio response generated: {len(content)} chars")
                return content.strip()
            else:
                logger.error(f"LM Studio API error {response.status_code}: {response.text}")
                return "Извините, произошла ошибка при генерации ответа."
                    
        except Exception as e:
            logger.error(f"Error in LM Studio client: {e}")
            return "Извините, не удалось получить ответ от ИИ модели."
    
    async def generate_with_context(self, 
                                  user_message: str,
                                  chat_history: List[Dict[str, str]] = None,
                                  system_prompt: str = None,
                                  max_tokens: int = 500,
                                  temperature: float = 0.7) -> str:
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            if chat_history:
                messages.extend(chat_history)
            
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return content.strip()
            else:
                logger.error(f"LM Studio API error {response.status_code}: {response.text}")
                return "Извините, произошла ошибка при генерации ответа."
                    
        except Exception as e:
            logger.error(f"Error in LM Studio client with context: {e}")
            return "Извините, не удалось получить ответ от ИИ модели."
    
    async def check_connection(self) -> bool:
        try:
            url = f"{self.base_url}/v1/models"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                logger.info(f"LM Studio connected. Available models: {models}")
                return True
            else:
                logger.error(f"LM Studio connection failed: {response.status_code}")
                return False
                    
        except Exception as e:
            logger.error(f"Error checking LM Studio connection: {e}")
            return False

class LMStudioClientSync:
    def __init__(self, base_url: str = "http://10.33.0.2:1234", model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, 
                        prompt: str, 
                        max_tokens: int = 500,
                        temperature: float = 0.7,
                        system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                logger.info(f"LM Studio response generated: {len(content)} chars")
                return content.strip()
            else:
                logger.error(f"LM Studio API error {response.status_code}: {response.text}")
                return "Извините, произошла ошибка при генерации ответа."
                
        except Exception as e:
            logger.error(f"Error in LM Studio client: {e}")
            return "Извините, не удалось получить ответ от ИИ модели."