import asyncio
import re
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, TOKENIZER


ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


class OCRInferenceEngine:
    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None
        self.processor = DeepseekOCRProcessor()
        self.model_path = MODEL_PATH

    async def initialize(self):
        """Initialize the vLLM engine"""
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def cleanup(self):
        """Cleanup resources"""
        # vLLM doesn't need explicit cleanup
        pass

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and correct image orientation"""
        try:
            image = Image.open(image_path)
            corrected_image = ImageOps.exif_transpose(image)
            return corrected_image
        except Exception as e:
            print(f"Error loading image: {e}")
            try:
                return Image.open(image_path)
            except:
                return None

    def re_match(self, text: str):
        """Extract reference patterns from text"""
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)

        matches_image = []
        matches_other = []
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])
        return matches, matches_image, matches_other

    def extract_coordinates_and_label(self, ref_text, image_width, image_height):
        """Extract coordinates and labels from reference text"""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
        except Exception as e:
            print(e)
            return None
        return (label_type, cor_list)

    def draw_bounding_boxes(self, image: Image.Image, refs, output_path: str):
        """Draw bounding boxes on image"""
        image_width, image_height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)

        font = ImageFont.load_default()

        img_idx = 0

        for i, ref in enumerate(refs):
            try:
                result = self.extract_coordinates_and_label(ref, image_width, image_height)
                if result:
                    label_type, points_list = result

                    color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                    color_a = color + (20,)

                    for points in points_list:
                        x1, y1, x2, y2 = points

                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)

                        if label_type == 'image':
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                cropped.save(f"{output_path}/images/{img_idx}.jpg")
                            except Exception as e:
                                print(e)
                                pass
                            img_idx += 1

                        try:
                            if label_type == 'title':
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            else:
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                            text_x = x1
                            text_y = max(0, y1 - 15)

                            text_bbox = draw.textbbox((0, 0), label_type, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                         fill=(255, 255, 255, 30))

                            draw.text((text_x, text_y), label_type, font=font, fill=color)
                        except:
                            pass
            except:
                continue

        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw

    async def stream_generate(self, image_features, prompt: str) -> str:
        """Run inference with streaming"""
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        request_id = f"request-{int(time.time())}-{np.random.randint(0, 10000)}"

        if image_features and '<image>' in prompt:
            request = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features}
            }
        elif prompt:
            request = {
                "prompt": prompt
            }
        else:
            raise ValueError('Prompt is required')

        full_text = ""
        async for request_output in self.engine.generate(
            request, sampling_params, request_id
        ):
            if request_output.outputs:
                full_text = request_output.outputs[0].text

        return full_text

    async def infer(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        min_crops: int = 2,
        max_crops: int = 6
    ) -> Dict[str, Any]:
        """Run complete inference pipeline"""

        # Load image
        image = self.load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = image.convert('RGB')

        # Process image
        if '<image>' in prompt:
            image_features = self.processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=crop_mode
            )
        else:
            image_features = ''

        # Run inference
        result_text = await self.stream_generate(image_features, prompt)

        # Save raw result
        raw_result_path = Path(output_path) / "result_ori.mmd"
        with open(raw_result_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        # Process results
        has_visualized_image = False
        processed_text = result_text

        if '<image>' in prompt:
            # Extract matches
            matches_ref, matches_images, matches_other = self.re_match(result_text)

            # Draw bounding boxes
            if matches_ref:
                result_image = self.draw_bounding_boxes(image, matches_ref, output_path)
                result_image_path = Path(output_path) / "result_with_boxes.jpg"
                result_image.save(result_image_path)
                has_visualized_image = True

            # Process text - replace image references
            for idx, a_match_image in enumerate(matches_images):
                processed_text = processed_text.replace(
                    a_match_image,
                    f'![](images/{idx}.jpg)\n'
                )

            # Remove other references
            for idx, a_match_other in enumerate(matches_other):
                processed_text = processed_text.replace(
                    a_match_other, ''
                ).replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        # Save processed result
        processed_result_path = Path(output_path) / "result.mmd"
        with open(processed_result_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)

        return {
            "raw_result": result_text,
            "processed_result": processed_text,
            "has_visualized_image": has_visualized_image
        }
