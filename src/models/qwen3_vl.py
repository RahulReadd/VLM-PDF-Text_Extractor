"""Adapter for Qwen3-VL family (2B / 4B / 8B).

Requires: pip install git+https://github.com/huggingface/transformers qwen-vl-utils
Uses Qwen3VLForConditionalGeneration + AutoProcessor.
"""

import torch
from PIL import Image

from .base import ModelConfig, VLMAdapter


class Qwen3VLAdapter(VLMAdapter):

    def load(self) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

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

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
