import requests
from io import BytesIO
from typing import Any, Dict, Optional

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

class LLaVARadModel:
    """
    Wrapper around microsoft/llava-rad that exposes a simple VQA interface:

        answer_vqa(image_path, system_prompt, user_prompt, **gen_kwargs) -> str
    """

    def __init__(self, cfg: Dict[str, Any]):
        disable_torch_init()

        self.model_path = cfg.get("model_path", "microsoft/llava-rad")
        self.model_base = cfg.get("model_base", "lmsys/vicuna-7b-v1.5")
        self.model_name = cfg.get("model_name", "llavarad")
        self.conv_mode = cfg.get("conv_mode", "v1")

        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = cfg.get("dtype", "bfloat16")  # or "bfloat16"/"float32"

        # Default generation settings (can be overridden per-call)
        gen_cfg = cfg.get("generation", {})
        self.default_temperature: float = gen_cfg.get("temperature", 0.0)
        self.default_max_new_tokens: int = gen_cfg.get("max_new_tokens", 512)
        self.default_do_sample: bool = gen_cfg.get("do_sample", False)

        # Load the pretrained LLaVA-Rad model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path,
            self.model_base,
            self.model_name,
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.image_processor = image_processor
        self.context_len = context_len

    def _build_prompt(
        self,
        user_prompt: str,
    ) -> str:
        """
        Build a LLaVA-style conversation prompt.
        """
        query = f"<image>\n{user_prompt.strip()}\n"

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def answer_vqa(
        self,
        image_path: str,
        user_prompt: str,
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
        prompt = self._build_prompt(user_prompt)

        # 2. Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.image_processor.preprocess(
            image,
            return_tensors="pt"
        )["pixel_values"][0]  # [C, H, W]

        # Move to device & dtype
        if self.dtype == "bfloat16":
            img_tensor = img_tensor.bfloat16()
        # else keep as float32

        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

        # 3. Tokenize text with image token
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)  # [1, T]

        # 4. Stopping criteria (stop at "</s>")
        stopping_criteria = KeywordsStoppingCriteria(
            stop_keywords=["</s>"],
            tokenizer=self.tokenizer,
            input_ids=input_ids,
        )

        # 5. Generation args (merge defaults with overrides)
        temperature = gen_kwargs.pop("temperature", self.default_temperature)
        max_new_tokens = gen_kwargs.pop("max_new_tokens", self.default_max_new_tokens)
        do_sample = gen_kwargs.pop("do_sample", self.default_do_sample)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=img_tensor,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **gen_kwargs,
            )

        # 6. Decode only the newly generated tokens
        new_tokens = output_ids[:, input_ids.shape[1]:]
        outputs = self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0]
        return outputs.strip()