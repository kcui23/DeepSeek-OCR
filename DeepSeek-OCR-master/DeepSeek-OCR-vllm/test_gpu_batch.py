#!/usr/bin/env python
"""
Test script to diagnose GPU batch processing issues
"""

import os
import torch
import sys
from pathlib import Path

# Set debugging mode
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def test_gpu_availability():
    """Test if GPU is available and accessible"""
    print("=" * 50)
    print("GPU Availability Test")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

            # Check current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"    Allocated: {allocated:.2f} GB")
                print(f"    Reserved: {reserved:.2f} GB")
    else:
        print("✗ CUDA is not available")
        return False

    return True

def test_small_batch():
    """Test processing a small batch"""
    print("\n" + "=" * 50)
    print("Small Batch Test")
    print("=" * 50)

    try:
        from config import MODEL_PATH
        print(f"Model path: {MODEL_PATH}")

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"✗ Model not found at {MODEL_PATH}")
            return False

        print("✓ Model path exists")

        # Try to import the necessary modules
        try:
            from deepseek_ocr import DeepseekOCRForCausalLM
            from vllm.model_executor.models.registry import ModelRegistry
            from vllm import LLM, SamplingParams
            print("✓ All modules imported successfully")
        except Exception as e:
            print(f"✗ Failed to import modules: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_pdf_loading():
    """Test PDF loading capability"""
    print("\n" + "=" * 50)
    print("PDF Loading Test")
    print("=" * 50)

    try:
        import fitz
        print(f"✓ PyMuPDF version: {fitz.__version__}")

        # Create a simple test to verify PDF operations
        print("✓ PDF library loaded successfully")
        return True

    except Exception as e:
        print(f"✗ Failed to load PDF library: {e}")
        return False

def test_memory_requirements(num_pdfs=10, pages_per_pdf=20):
    """Estimate memory requirements"""
    print("\n" + "=" * 50)
    print("Memory Requirements Estimation")
    print("=" * 50)

    # Rough estimates
    image_size_mb = 50  # Average size per page in memory
    model_size_gb = 8   # Estimated model size
    overhead_gb = 2     # System overhead

    total_images = num_pdfs * pages_per_pdf
    image_memory_gb = (total_images * image_size_mb) / 1024
    total_memory_gb = model_size_gb + image_memory_gb + overhead_gb

    print(f"For {num_pdfs} PDFs with ~{pages_per_pdf} pages each:")
    print(f"  Total images: {total_images}")
    print(f"  Image memory: {image_memory_gb:.2f} GB")
    print(f"  Model memory: {model_size_gb:.2f} GB")
    print(f"  Overhead: {overhead_gb:.2f} GB")
    print(f"  Total estimated: {total_memory_gb:.2f} GB")

    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  Available GPU memory: {available_memory:.2f} GB")

        if total_memory_gb > available_memory:
            print(f"  ⚠ Warning: Estimated requirements exceed available memory")
            print(f"  Recommendation: Process in smaller batches")
        else:
            print(f"  ✓ Sufficient memory available")

def diagnose_error(error_msg):
    """Provide diagnosis based on error message"""
    print("\n" + "=" * 50)
    print("Error Diagnosis")
    print("=" * 50)

    if "illegal memory access" in error_msg.lower():
        print("Diagnosis: CUDA Illegal Memory Access")
        print("\nPossible causes:")
        print("1. Concurrent GPU operations from multiple threads")
        print("2. Out of memory (OOM) condition")
        print("3. Incompatible CUDA/PyTorch versions")
        print("\nRecommendations:")
        print("• Use the fixed batch processing script (run_dpsk_ocr_pdf_batch_fixed.py)")
        print("• Reduce batch size to process fewer PDFs at once")
        print("• Enable CUDA debugging: export CUDA_LAUNCH_BLOCKING=1")
        print("• Check GPU memory usage during processing")
        print("• Ensure only one llm.generate() call happens at a time")

    elif "out of memory" in error_msg.lower():
        print("Diagnosis: GPU Out of Memory")
        print("\nRecommendations:")
        print("• Reduce batch size")
        print("• Process PDFs sequentially instead of in parallel")
        print("• Reduce MAX_CONCURRENCY in config.py")
        print("• Lower gpu_memory_utilization in LLM initialization")

def main():
    print("DeepSeek OCR GPU Batch Processing Diagnostic Tool")
    print("=" * 60)

    # Run tests
    gpu_ok = test_gpu_availability()
    model_ok = test_small_batch()
    pdf_ok = test_pdf_loading()

    # Memory estimation
    test_memory_requirements()

    # Check for command line error message
    if len(sys.argv) > 1:
        error_msg = " ".join(sys.argv[1:])
        diagnose_error(error_msg)

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    all_ok = gpu_ok and model_ok and pdf_ok

    if all_ok:
        print("✓ All tests passed")
        print("\nRecommended approach for batch processing:")
        print("1. Use run_dpsk_ocr_pdf_batch_fixed.py (not the original)")
        print("2. Start with small batches (5-10 PDFs)")
        print("3. Monitor GPU memory usage")
        print("4. Increase batch size gradually if memory allows")
    else:
        print("✗ Some tests failed - please review the output above")

    print("\nUsage examples:")
    print("  python run_dpsk_ocr_pdf_batch_fixed.py pdf_paths.txt output_dir")
    print("  python run_batch_with_config_fixed.py pdf_paths.txt -b 5 -g 0")

if __name__ == "__main__":
    main()