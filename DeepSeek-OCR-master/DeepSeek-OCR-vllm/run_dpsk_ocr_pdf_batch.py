import os
import fitz
import img2pdf
import io
import re
import json
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from config import MODEL_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Initialize the LLM model globally
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """Convert PDF to images"""
    images = []

    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

        images.append(img)

    pdf_document.close()
    return images

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other

def extract_image_captions(content):
    """
    Extract image and corresponding caption from content
    Returns a dictionary where key is image index and value is caption text
    Only includes images that are followed by image_caption
    """
    image_caption_map = {}

    # Match all <|ref|>image<|/ref|><|det|>...<|/det|>
    image_pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>.*?<\|/det\|>'

    # Find all image markers and their positions
    image_matches = list(re.finditer(image_pattern, content, re.DOTALL))

    for idx, img_match in enumerate(image_matches):
        # Get the end position of current image marker
        img_end_pos = img_match.end()

        # From the end position of image marker, find the following content
        remaining_text = content[img_end_pos:]

        # Check if followed by image_caption (allowing whitespace)
        caption_pattern = r'^\s*<\|ref\|>image_caption<\|/ref\|><\|det\|>.*?<\|/det\|>\s*<center>(.*?)</center>'
        caption_match = re.match(caption_pattern, remaining_text, re.DOTALL)

        if caption_match:
            caption_text = caption_match.group(1).strip()
            if caption_text:
                image_caption_map[idx] = caption_text

    return image_caption_map

def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)

def process_single_image(image, prompt):
    """Process single image"""
    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item

def process_single_pdf(pdf_path, base_output_path):
    """
    Process a single PDF file and save results to a dedicated folder
    """
    try:
        pdf_name = Path(pdf_path).stem
        pdf_output_dir = os.path.join(base_output_path, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        print(f'{Colors.BLUE}Processing PDF: {pdf_path}{Colors.RESET}')

        # Convert PDF to images
        images = pdf_to_images_high_quality(pdf_path)

        if not images:
            print(f'{Colors.YELLOW}No images found in PDF: {pdf_path}{Colors.RESET}')
            return

        # Prepare batch inputs
        prompt = PROMPT
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(lambda img: process_single_image(img, prompt), images),
                total=len(images),
                desc=f"Pre-processing images for {pdf_name}"
            ))

        # Generate outputs using OCR model
        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )

        # Process outputs and extract images with captions
        image_caption_dict = {}

        for page_idx, (output, img) in enumerate(zip(outputs_list, images)):
            content = output.outputs[0].text

            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
            else:
                if SKIP_REPEAT:
                    continue

            # Extract matches and image captions
            matches_ref, matches_images, mathes_other = re_match(content)

            # Extract image_caption mappings for this page
            page_captions = extract_image_captions(content)

            # Extract and save only images with captions
            image_width, image_height = img.size

            img_idx = 0
            for i, ref in enumerate(matches_ref):
                try:
                    result = extract_coordinates_and_label(ref, image_width, image_height)
                    if result:
                        label_type, points_list = result

                        if label_type == 'image':
                            # Check if this image has a caption
                            if img_idx in page_captions:
                                for points in points_list:
                                    x1, y1, x2, y2 = points

                                    x1 = int(x1 / 999 * image_width)
                                    y1 = int(y1 / 999 * image_height)
                                    x2 = int(x2 / 999 * image_width)
                                    y2 = int(y2 / 999 * image_height)

                                    try:
                                        # Save image only if it has a caption
                                        cropped = img.crop((x1, y1, x2, y2))
                                        image_filename = f"page_{page_idx}_image_{img_idx}.jpg"
                                        image_path = os.path.join(pdf_output_dir, image_filename)
                                        cropped.save(image_path)

                                        # Add to caption dictionary
                                        image_caption_dict[image_filename] = page_captions[img_idx]
                                    except Exception as e:
                                        print(f"Error saving image: {e}")

                            img_idx += 1
                except Exception as e:
                    continue

        # Save image_captions.json
        if image_caption_dict:
            json_caption_path = os.path.join(pdf_output_dir, 'image_captions.json')
            with open(json_caption_path, 'w', encoding='utf-8') as f:
                json.dump(image_caption_dict, f, ensure_ascii=False, indent=2)

            print(f'{Colors.GREEN}Saved {len(image_caption_dict)} image-caption pairs for {pdf_name}{Colors.RESET}')
        else:
            print(f'{Colors.YELLOW}No images with captions found in {pdf_name}{Colors.RESET}')

        return pdf_name, len(image_caption_dict)

    except Exception as e:
        print(f'{Colors.RED}Error processing {pdf_path}: {str(e)}{Colors.RESET}')
        return pdf_path, 0

def process_pdfs_concurrently(pdf_paths, output_path, max_workers=10):
    """
    Process multiple PDFs concurrently with a maximum number of workers
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all PDF processing tasks
        futures = [
            executor.submit(process_single_pdf, pdf_path, output_path)
            for pdf_path in pdf_paths
        ]

        # Collect results with progress bar
        for future in tqdm(futures, desc="Processing PDFs"):
            result = future.result()
            if result:
                results.append(result)

    return results

def main(input_txt_file, output_dir):
    """
    Main function to process multiple PDFs from a text file

    Args:
        input_txt_file: Path to text file containing PDF paths (one per line)
        output_dir: Base output directory for all processed PDFs
    """
    # Read PDF paths from text file
    if not os.path.exists(input_txt_file):
        print(f'{Colors.RED}Input file not found: {input_txt_file}{Colors.RESET}')
        return

    with open(input_txt_file, 'r', encoding='utf-8') as f:
        pdf_paths = [line.strip() for line in f if line.strip()]

    # Filter out non-existent PDF files
    valid_pdf_paths = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            valid_pdf_paths.append(pdf_path)
        else:
            print(f'{Colors.YELLOW}PDF file not found: {pdf_path}{Colors.RESET}')

    if not valid_pdf_paths:
        print(f'{Colors.RED}No valid PDF files found in input file{Colors.RESET}')
        return

    print(f'{Colors.GREEN}Found {len(valid_pdf_paths)} valid PDF files to process{Colors.RESET}')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process PDFs concurrently
    results = process_pdfs_concurrently(valid_pdf_paths, output_dir, max_workers=10)

    # Summary
    print(f'\n{Colors.GREEN}Processing complete!{Colors.RESET}')
    print(f'Total PDFs processed: {len(results)}')

    total_images = sum(count for _, count in results)
    print(f'Total images with captions saved: {total_images}')

    for pdf_name, count in results:
        print(f'  - {pdf_name}: {count} images')

if __name__ == "__main__":
    import sys

    # Default values if not provided
    if len(sys.argv) < 2:
        print(f'{Colors.YELLOW}Usage: python run_dpsk_ocr_pdf_batch.py <input_txt_file> [output_directory]{Colors.RESET}')
        print(f'{Colors.YELLOW}Using default values from config.py{Colors.RESET}')

        # Create a sample input text file for testing
        sample_input_file = "pdf_paths.txt"
        if not os.path.exists(sample_input_file):
            print(f'{Colors.RED}Please create a file named "pdf_paths.txt" with PDF paths (one per line){Colors.RESET}')
            sys.exit(1)

        input_txt_file = sample_input_file
        output_dir = OUTPUT_PATH
    else:
        input_txt_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    main(input_txt_file, output_dir)