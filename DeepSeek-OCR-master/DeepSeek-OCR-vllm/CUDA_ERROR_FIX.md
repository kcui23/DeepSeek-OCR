# Fix for CUDA Illegal Memory Access Error

## Problem Description

When running the original batch PDF processing script with concurrent PDF processing, you may encounter:

```
Error processing /path/to/file.pdf: CUDA error: an illegal memory access was encountered
```

This error occurs because the VLLM model cannot handle multiple concurrent `llm.generate()` calls from different threads. Each thread trying to access the GPU simultaneously causes memory conflicts.

## Root Cause

The original implementation (`run_dpsk_ocr_pdf_batch.py`) attempted to:
1. Process multiple PDFs concurrently using ThreadPoolExecutor
2. Each PDF processing thread called `llm.generate()` independently
3. Multiple simultaneous GPU operations caused memory access violations

## Solution

The fixed implementation (`run_dpsk_ocr_pdf_batch_fixed.py`) follows the pattern from `run_dpsk_ocr_eval_batch.py`:

### Key Changes:

1. **Batch Collection First**
   - Load all PDFs and collect all images with metadata
   - Maintain mapping of which image belongs to which PDF

2. **Single GPU Operation**
   - Pre-process all images concurrently (CPU operation)
   - Make ONE single `llm.generate()` call for all images
   - No concurrent GPU access

3. **Result Distribution**
   - After getting all results, distribute them back to respective PDFs
   - Save images and captions to appropriate folders

## Architecture Comparison

### Original (Causes Error):
```
PDF1 ─┐
PDF2 ─┼─> Concurrent Processing ─> Multiple llm.generate() ─> CUDA ERROR
PDF3 ─┘   (each calls GPU)         (concurrent GPU access)
```

### Fixed:
```
PDF1 ─┐
PDF2 ─┼─> Load All ─> Pre-process ─> Single llm.generate() ─> Distribute
PDF3 ─┘   (Sequential)  (Concurrent)    (One GPU call)        Results
```

## Usage

### Use the Fixed Version:
```bash
# Basic usage
python run_dpsk_ocr_pdf_batch_fixed.py pdf_paths.txt output_dir

# With configuration
python run_batch_with_config_fixed.py pdf_paths.txt -o output -b 10 -g 0
```

### DO NOT Use:
```bash
# This will cause CUDA errors
python run_dpsk_ocr_pdf_batch.py pdf_paths.txt output_dir
```

## Memory Management

The fixed version processes PDFs in configurable batches to manage memory:

- Default: 10 PDFs per batch
- Adjustable via `-b` flag in config script
- Each batch is processed completely before moving to next

## Performance Considerations

1. **Memory Usage**: All images from batch PDFs are loaded into memory
2. **GPU Memory**: Single large batch may use more GPU memory
3. **Speed**: Actually faster due to better GPU utilization

## Debugging

If you still encounter issues:

1. **Enable CUDA debugging**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python run_dpsk_ocr_pdf_batch_fixed.py pdf_paths.txt output_dir
   ```

2. **Run diagnostic tool**:
   ```bash
   python test_gpu_batch.py
   ```

3. **Reduce batch size**:
   ```bash
   python run_batch_with_config_fixed.py pdf_paths.txt -b 5
   ```

## Key Takeaway

**VLLM models must have exclusive GPU access during generation.** Never call `llm.generate()` from multiple threads simultaneously. Always batch all inputs and make a single generation call.