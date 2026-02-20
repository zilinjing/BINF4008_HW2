import requests
from io import BytesIO
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

class CheXagentModel:
    """
    Wrapper around chexagent that exposes a simple VQA interface:

        answer_vqa(image_path, system_prompt, user_prompt, **gen_kwargs) -> str
    """

    def __init__(self, cfg: Dict[str, Any]):

        self.model_name = cfg.get("model_name", "StanfordAIMI/CheXagent-2-3b")

        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = cfg.get("dtype", "bfloat16")  # or "bfloat16"/"float32"

        if dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Default generation settings (can be overridden per-call)
        gen_cfg = cfg.get("generation", {})
        self.default_temperature: float = gen_cfg.get("temperature", 0.0)
        self.default_max_new_tokens: int = gen_cfg.get("max_new_tokens", 512)
        self.default_do_sample: bool = gen_cfg.get("do_sample", False)

        # Load the pretrained LLaVA-Rad model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True)
        model = model.to(self.dtype)
        model.eval()

        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def _build_messages(
        self,
        user_prompt: str,
        image: Image.Image,
    ) -> str:
        """
        Build the conversation prompt.
        """
        query = self.tokenizer.from_list_format([*[{'image': image}], {'text': user_prompt}])
        conv = [{"from": "system", "value": "You are an expert radiologist."}, {"from": "human", "value": query}]
        return conv
    
    def answer_vqa_batch(
        self,
        image_paths: str,
        user_prompts: str,
        **gen_kwargs: Any,
    ) -> str:
        """
        Main entry point used by your runner.

        Args:
            image_path: path or URL to the chest X-ray.
            system_prompt: optional system-level instruction (may be ignored for LLaVA).
            user_prompt: the actual question / instruction about the image.
            gen_kwargs: optional overrides for temperature, max_new_tokens, etc.

        Returns:
            A string answer from the model.
        """
        # 1. Build the conversation prompt
        temperature = gen_kwargs.pop("temperature", self.default_temperature)
        max_new_tokens = gen_kwargs.pop("max_new_tokens", self.default_max_new_tokens)
        do_sample = gen_kwargs.pop("do_sample", self.default_do_sample)

        conversations = []
        for img, prompt in zip(image_paths, user_prompts):
            conv = self._build_messages(prompt, img)  # your existing helper
            conversations.append(conv)

        input_ids = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        )

        input_lengths = int(input_ids.shape[1])

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids.to(self.device),
                do_sample=do_sample, 
                temperature=temperature, 
                use_cache=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        outputs = []
        for i in range(output.size(0)):
            start = input_lengths
            new_tokens = output[i, start:]
            text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            outputs.append(text.strip())

        return outputs