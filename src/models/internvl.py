"""Adapter for InternVL 2.5 / 3.5 family (2B variants).

Requires: pip install git+https://github.com/huggingface/transformers
Uses AutoModelForImageTextToText + AutoProcessor (HF unified API).
"""

import torch
from PIL import Image

from .base import ModelConfig, VLMAdapter


class InternVLAdapter(VLMAdapter):

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        load_kwargs: dict = dict(
            torch_dtype=self.config.dtype,
            device_map=self.config.device_map,
        )

        if self.config.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

    def run_inference(self, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
