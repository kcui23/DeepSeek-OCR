#!/usr/bin/env python
"""
Batch PDF processing with configurable parameters - Fixed version
"""

import argparse
import os
import sys
from pathlib import Path

# Import the fixed processing function
from run_dpsk_ocr_pdf_batch_fixed import main, Colors

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch process PDFs with DeepSeek OCR (Fixed GPU concurrency)",
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
        "-b", "--batch-size",
        type=int,
        default=10,
        help="Number of PDFs to process in each batch (helps manage memory)"
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable CUDA debugging mode (slower but helps identify errors)"
    )

    return parser.parse_args()

def validate_input_file(input_file):
    """Validate input file and return list of valid PDF paths"""
    if not os.path.exists(input_file):
        print(f"{Colors.RED}Error: Input file not found: {input_file}{Colors.RESET}")
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

    # Enable CUDA debugging if requested
    if args.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print(f"{Colors.YELLOW}CUDA debugging mode enabled (may run slower){Colors.RESET}")

    # Validate input file
    print(f"{Colors.BLUE}Validating input file: {args.input_file}{Colors.RESET}")
    result = validate_input_file(args.input_file)

    if result is None:
        sys.exit(1)

    valid_pdfs, invalid_pdfs = result

    # Report validation results
    print(f"\n{Colors.GREEN}Validation Results:{Colors.RESET}")
    print(f"  Valid PDFs found: {len(valid_pdfs)}")
    print(f"  Invalid entries: {len(invalid_pdfs)}")

    if invalid_pdfs:
        print(f"\n{Colors.YELLOW}Invalid entries:{Colors.RESET}")
        for line_num, path, reason in invalid_pdfs[:10]:  # Show first 10
            print(f"  Line {line_num}: {reason} - {path}")
        if len(invalid_pdfs) > 10:
            print(f"  ... and {len(invalid_pdfs) - 10} more")

    if not valid_pdfs:
        print(f"\n{Colors.RED}No valid PDFs found to process.{Colors.RESET}")
        sys.exit(1)

    print(f"\n{Colors.GREEN}PDFs to process: {len(valid_pdfs)}{Colors.RESET}")
    for pdf in valid_pdfs[:5]:  # Show first 5
        print(f"  - {Path(pdf).name}")
    if len(valid_pdfs) > 5:
        print(f"  ... and {len(valid_pdfs) - 5} more")

    # Calculate memory requirements estimate
    estimated_pages = len(valid_pdfs) * 20  # Assume average 20 pages per PDF
    estimated_memory_gb = estimated_pages * 0.05  # Rough estimate: 50MB per page
    print(f"\n{Colors.YELLOW}Estimated memory requirement: ~{estimated_memory_gb:.1f} GB{Colors.RESET}")
    print(f"Processing in batches of {args.batch_size} PDFs")

    # Dry run mode
    if args.dry_run:
        print(f"\n{Colors.YELLOW}[DRY RUN MODE] No processing will be performed.{Colors.RESET}")
        print(f"Output would be saved to: {args.output_dir}")
        print(f"Batch size: {args.batch_size} PDFs per batch")
        return

    # Confirm before processing
    response = input(f"\n{Colors.BLUE}Proceed with processing? (y/n): {Colors.RESET}")
    if response.lower() != 'y':
        print(f"{Colors.YELLOW}Processing cancelled.{Colors.RESET}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Update batch size in the processing module
    import run_dpsk_ocr_pdf_batch_fixed
    original_batch_size = getattr(run_dpsk_ocr_pdf_batch_fixed, 'BATCH_SIZE', 10)
    run_dpsk_ocr_pdf_batch_fixed.BATCH_SIZE = args.batch_size

    # Process the PDFs
    print(f"\n{Colors.GREEN}Starting batch processing...{Colors.RESET}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size} PDFs per batch")
    print(f"GPU device: {args.gpu}")

    try:
        # Run main processing
        main(args.input_file, args.output_dir)
    except Exception as e:
        print(f"\n{Colors.RED}Error during processing: {str(e)}{Colors.RESET}")
        if "CUDA" in str(e):
            print(f"{Colors.YELLOW}Tip: Try reducing batch size with -b flag or enable debug mode with --debug{Colors.RESET}")
        sys.exit(1)
    finally:
        # Restore original batch size
        run_dpsk_ocr_pdf_batch_fixed.BATCH_SIZE = original_batch_size

    print(f"\n{Colors.GREEN}Processing completed successfully!{Colors.RESET}")

if __name__ == "__main__":
    main_cli()