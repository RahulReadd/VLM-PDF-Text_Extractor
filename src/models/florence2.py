"""Adapter for Microsoft Florence-2 family (0.77B).

Florence-2 uses task-specific tokens (<OCR>, <OCR_WITH_REGION>, etc.) rather than
free-form prompts.  For structured extraction we send the user prompt as-is and
let the model treat it as a VQA/caption task, which still produces usable text.

Requires: pip install git+https://github.com/huggingface/transformers
"""

import torch
from PIL import Image

from .base import ModelConfig, VLMAdapter


class Florence2Adapter(VLMAdapter):

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        load_kwargs: dict = dict(
            torch_dtype=self.config.dtype,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )

    def run_inference(self, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
        # Florence-2 expects a task token or plain text as the prompt.
        # For structured extraction we pass the user's prompt directly.
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model.device, dtype=self.config.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # post_process_generation is optional; strip task tokens for clean output
        cleaned = decoded.replace("<s>", "").replace("</s>", "").strip()
        return cleaned
