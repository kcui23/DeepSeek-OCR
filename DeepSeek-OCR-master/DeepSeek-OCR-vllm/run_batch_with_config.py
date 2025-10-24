#!/usr/bin/env python
"""
Batch PDF processing with configurable parameters
"""

import argparse
import os
import sys
from pathlib import Path

# Import the main processing function
from run_dpsk_ocr_pdf_batch import main

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch process PDFs with DeepSeek OCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_file",
        help="Text file containing PDF paths (one per line)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Output directory for processed PDFs"
    )

    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=10,
        help="Maximum number of PDFs to process concurrently"
    )

    parser.add_argument(
        "-g", "--gpu",
        type=str,
        default="7",
        help="GPU device ID to use (CUDA_VISIBLE_DEVICES)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input file and show what would be processed"
    )

    return parser.parse_args()

def validate_input_file(input_file):
    """Validate input file and return list of valid PDF paths"""
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return None

    valid_pdfs = []
    invalid_pdfs = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Check if file exists and is a PDF
            if os.path.exists(line):
                if line.lower().endswith('.pdf'):
                    valid_pdfs.append(line)
                else:
                    invalid_pdfs.append((line_num, line, "Not a PDF file"))
            else:
                invalid_pdfs.append((line_num, line, "File not found"))

    return valid_pdfs, invalid_pdfs

def main_cli():
    args = parse_arguments()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Validate input file
    print(f"Validating input file: {args.input_file}")
    result = validate_input_file(args.input_file)

    if result is None:
        sys.exit(1)

    valid_pdfs, invalid_pdfs = result

    # Report validation results
    print(f"\nValidation Results:")
    print(f"  Valid PDFs found: {len(valid_pdfs)}")
    print(f"  Invalid entries: {len(invalid_pdfs)}")

    if invalid_pdfs:
        print("\nInvalid entries:")
        for line_num, path, reason in invalid_pdfs[:10]:  # Show first 10
            print(f"  Line {line_num}: {reason} - {path}")
        if len(invalid_pdfs) > 10:
            print(f"  ... and {len(invalid_pdfs) - 10} more")

    if not valid_pdfs:
        print("\nNo valid PDFs found to process.")
        sys.exit(1)

    print(f"\nPDFs to process: {len(valid_pdfs)}")
    for pdf in valid_pdfs[:5]:  # Show first 5
        print(f"  - {Path(pdf).name}")
    if len(valid_pdfs) > 5:
        print(f"  ... and {len(valid_pdfs) - 5} more")

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE] No processing will be performed.")
        print(f"Output would be saved to: {args.output_dir}")
        print(f"Max concurrent workers: {args.max_workers}")
        return

    # Confirm before processing
    response = input("\nProceed with processing? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the PDFs
    print(f"\nStarting batch processing with {args.max_workers} workers...")
    print(f"Output directory: {args.output_dir}")

    # Temporarily update the max_workers in the imported module
    import run_dpsk_ocr_pdf_batch
    original_process_func = run_dpsk_ocr_pdf_batch.process_pdfs_concurrently

    def custom_process_func(pdf_paths, output_path, max_workers=10):
        return original_process_func(pdf_paths, output_path, args.max_workers)

    run_dpsk_ocr_pdf_batch.process_pdfs_concurrently = custom_process_func

    # Run main processing
    main(args.input_file, args.output_dir)

if __name__ == "__main__":
    main_cli()