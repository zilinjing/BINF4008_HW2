from io import BytesIO
from typing import Any, Dict, Optional

import requests
from PIL import Image
# from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm import LLM, SamplingParams

class MedGemmaModel:
    """
    Wrapper around google/medgemma-* models that exposes a VQA-style interface:
        answer_vqa(image_path, system_prompt, user_prompt, **gen_kwargs) -> str
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.model_id: str = cfg.get("model_id", "google/medgemma-4b-it")
        # self.device_map: str | Dict[str, int] = cfg.get("device_map", "auto")
        # dtype_str: str = cfg.get("torch_dtype", "bfloat16")

        # if dtype_str == "bfloat16":
        #     torch_dtype = torch.bfloat16
        # else:
        #     torch_dtype = torch.float32

        # gen_cfg = cfg.get("generation", {})
        # self.default_max_new_tokens: int = gen_cfg.get("max_new_tokens", 512)
        # self.default_temperature: float = gen_cfg.get("temperature", 0.0)
        # self.default_do_sample: bool = gen_cfg.get("do_sample", False)

        # # Load model + processor
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     self.model_id,
        #     torch_dtype=torch_dtype,
        #     device_map=self.device_map,
        # )
        # self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.torch_dtype = torch_dtype

        engine_cfg = cfg.get("engine", {})
        dtype = engine_cfg.get("dtype", "bfloat16")
        gpu_memory_utilization = engine_cfg.get("gpu_memory_utilization", 0.9)
        max_model_len = engine_cfg.get("max_model_len", 5000)

        # Initialize vLLM engine
        self.llm = LLM(
            model=self.model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        gen_cfg = cfg.get("generation", {})
        self.default_max_new_tokens: int = gen_cfg.get("max_new_tokens", 512)
        self.default_temperature: float = gen_cfg.get("temperature", 0.0)
        self.default_top_p: float = gen_cfg.get("top_p", 1.0)
        self.default_top_k: int = gen_cfg.get("top_k", -1)

    def _build_sampling_params(self, overrides: Dict[str, Any]) -> SamplingParams:
        max_tokens = overrides.pop("max_new_tokens", self.default_max_new_tokens)
        temperature = overrides.pop("temperature", self.default_temperature)
        top_p = overrides.pop("top_p", self.default_top_p)
        top_k = overrides.pop("top_k", self.default_top_k)

        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


    def _build_messages(
        self,
        user_prompt: str,
        image: Image.Image,
    ):
        """
        Build MedGemma-style chat messages with image + question.
        """
        sys_text = "You are an expert radiologist."
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_pil", "image_pil": image},
                ],
            },
        ]
        return messages
    
    def answer_vqa(
        self,
        image_path: str,
        user_prompt: str,
        **gen_kwargs: Any,
    ) -> str:
        """
        Args:
            image_path: local path or URL to chest X-ray.
            system_prompt: optional system-level instruction.
            user_prompt: question about the image.
            gen_kwargs: overrides for generation (temperature, max_new_tokens, etc.).

        Returns:
            Decoded answer string.
        """
        # 1. Load image
        image = Image.open(image_path).convert("RGB")

        # 2. Build messages for MedGemma chat template
        messages = self._build_messages(user_prompt, image)

        sampling_params = self._build_sampling_params(gen_kwargs)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        decoded = outputs[0].outputs[0].text

        return decoded.strip()
    
    def answer_vqa_batch(
        self,
        image_paths: list[str],
        user_prompts: list[str],
        **gen_kwargs: Any,
    ) -> list[str]:
        """
        Batched VQA interface.

        Args:
            image_paths: list of local paths or URLs to chest X-rays.
            user_prompts: list of questions (same length as image_paths).
            gen_kwargs: overrides for generation (temperature, max_new_tokens, etc.).

        Returns:
            List of decoded answers, one per (image, prompt) pair.
        """
        assert len(image_paths) == len(user_prompts), "Batch: len(image_paths) != len(user_prompts)"

        # 1. Load all images as PIL
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # 2. Build conversations list for vLLM.chat
        conversations = []
        for img, prompt in zip(images, user_prompts):
            conv = self._build_messages(prompt, img)   # your existing helper
            conversations.append(conv)

        # 3. Sampling params (same for whole batch)
        sampling_params = self._build_sampling_params(gen_kwargs)

        # 4. Run vLLM chat in batch
        outputs = self.llm.chat(conversations, sampling_params=sampling_params)

        # 5. Extract text per sample
        preds: list[str] = []
        for o in outputs:
            text = o.outputs[0].text.strip()
            preds.append(text)
        return preds