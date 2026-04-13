"""Adapter for Microsoft Florence-2 family (0.77B).

Florence-2 uses task-specific tokens (<OCR>, <OCR_WITH_REGION>, etc.) rather than
free-form chat prompts.  We map our generic prompts to Florence's <OCR> task and
wrap the raw OCR text into a JSON object so the rest of the pipeline can parse it.

Requires: pip install git+https://github.com/huggingface/transformers
"""

import json

import torch
from PIL import Image

from .base import ModelConfig, VLMAdapter


def _patch_florence2_config_class():
    """Monkey-patch Florence-2's remote config class BEFORE from_pretrained.

    The remote Florence2LanguageConfig is missing `forced_bos_token_id`,
    which newer transformers accesses via __getattribute__ during loading.
    We intercept the config's __init__ to inject the missing default.
    """
    from transformers import AutoConfig  # type: ignore[import-unresolved]

    config = AutoConfig.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )

    for cfg_attr in ("text_config", "language_config"):
        cfg = getattr(config, cfg_attr, None)
        if cfg is not None:
            cfg_cls = type(cfg)
            if not hasattr(cfg_cls, "_f2_patched"):
                _original_init = cfg_cls.__init__

                def _patched_init(self, *args, _orig=_original_init, **kwargs):
                    _orig(self, *args, **kwargs)
                    if not hasattr(self, "forced_bos_token_id"):
                        self.forced_bos_token_id = None

                cfg_cls.__init__ = _patched_init
                cfg_cls._f2_patched = True

    return config


class Florence2Adapter(VLMAdapter):

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-unresolved]

        config = _patch_florence2_config_class()

        load_kwargs: dict = dict(
            torch_dtype=self.config.dtype,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, config=config, **load_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )

    def run_inference(self, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
        # Florence-2 requires a task token, not free-form text.
        # Detect if prompt already is a task token; otherwise default to <OCR>.
        if prompt.startswith("<") and prompt.endswith(">"):
            task_token = prompt
        else:
            task_token = "<OCR>"

        inputs = self.processor(
            text=task_token, images=image, return_tensors="pt"
        ).to(self.model.device, dtype=self.config.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
                early_stopping=False,
            )

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed = self.processor.post_process_generation(
            decoded,
            task=task_token,
            image_size=(image.width, image.height),
        )

        # parsed is a dict like {"<OCR>": "extracted text here"}
        ocr_text = parsed.get(task_token, decoded)

        # Wrap raw OCR text into a JSON object so downstream JSON parsing succeeds
        return json.dumps({"ocr_text": ocr_text})
