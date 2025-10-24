#!/bin/bash

# Safe batch processing script for DeepSeek OCR
# This script uses the fixed version that avoids CUDA errors

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}DeepSeek OCR Safe Batch Processing${NC}"
echo "======================================"

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No input file provided${NC}"
    echo "Usage: $0 <pdf_list.txt> [output_dir] [batch_size] [gpu_id]"
    echo ""
    echo "Example:"
    echo "  $0 pdf_paths.txt output 10 0"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_DIR=${2:-"output"}
BATCH_SIZE=${3:-10}
GPU_ID=${4:-0}

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Count valid PDFs
PDF_COUNT=$(grep -v '^#' "$INPUT_FILE" | grep -v '^$' | wc -l)
echo -e "${GREEN}Found $PDF_COUNT PDFs in input file${NC}"

# Estimate processing time (rough estimate: 30 seconds per PDF)
ESTIMATED_TIME=$((PDF_COUNT * 30 / 60))
echo -e "${YELLOW}Estimated processing time: ~$ESTIMATED_TIME minutes${NC}"

# Set environment variables for safety
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_LAUNCH_BLOCKING=1  # Enable debugging for safety

echo ""
echo "Configuration:"
echo "  Input file: $INPUT_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE PDFs per batch"
echo "  GPU ID: $GPU_ID"
echo ""

# Ask for confirmation
read -p "Proceed with processing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Processing cancelled${NC}"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the fixed batch processor
echo -e "${GREEN}Starting batch processing...${NC}"
python run_dpsk_ocr_pdf_batch_fixed.py "$INPUT_FILE" "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Processing completed successfully!${NC}"

    # Count results
    IMAGE_COUNT=$(find "$OUTPUT_DIR" -name "*.jpg" | wc -l)
    JSON_COUNT=$(find "$OUTPUT_DIR" -name "image_captions.json" | wc -l)

    echo ""
    echo "Results summary:"
    echo "  PDFs processed: $JSON_COUNT"
    echo "  Images extracted: $IMAGE_COUNT"
    echo "  Output location: $OUTPUT_DIR"
else
    echo -e "${RED}✗ Processing failed with errors${NC}"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check GPU memory: nvidia-smi"
    echo "2. Try smaller batch size: $0 $INPUT_FILE $OUTPUT_DIR 5"
    echo "3. Run diagnostic: python test_gpu_batch.py"
    echo "4. Check error log above for details"
    exit 1
fi