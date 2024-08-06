from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

@dataclass
class ModelConfig:
    def __init__(self, tokenizer: AutoTokenizer, gen_params: Optional[dict]=None, model_name: Optional[str]="",
                 use_chat_template: Optional[bool]=True, chat_template_type: str="system_user", device: str="cuda"):
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.chat_template_type = chat_template_type
        self.gen_params = gen_params
        self.model_name = model_name
        self.device = device