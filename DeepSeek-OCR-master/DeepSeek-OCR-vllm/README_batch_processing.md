# DeepSeek OCR Batch PDF Processing

This updated version of the DeepSeek OCR system provides batch processing capabilities for multiple PDF files with concurrent execution.

## Key Features

- **Batch Processing**: Process multiple PDFs from a text file input
- **Concurrent Execution**: Process up to 10 PDFs simultaneously for faster throughput
- **Organized Output**: Each PDF gets its own output folder
- **Caption-Based Filtering**: Only saves images that have associated captions
- **JSON Caption Mapping**: Generates `image_captions.json` for each PDF with image-caption pairs

## File Structure

```
output_directory/
├── pdf_name_1/
│   ├── page_0_image_0.jpg
│   ├── page_0_image_1.jpg
│   ├── page_1_image_0.jpg
│   └── image_captions.json
├── pdf_name_2/
│   ├── page_0_image_0.jpg
│   ├── page_2_image_0.jpg
│   └── image_captions.json
└── ...
```

## Usage

### Method 1: Direct Script Execution

```bash
python run_dpsk_ocr_pdf_batch.py <input_txt_file> [output_directory]
```

Example:
```bash
python run_dpsk_ocr_pdf_batch.py pdf_paths.txt ./output
```

### Method 2: With Configuration Options

```bash
python run_batch_with_config.py input_file.txt -o output_dir -w 10 -g 7
```

Options:
- `-o, --output-dir`: Output directory (default: ./output)
- `-w, --max-workers`: Maximum concurrent PDFs to process (default: 10)
- `-g, --gpu`: GPU device ID (default: 7)
- `--dry-run`: Validate input without processing

### Input File Format

Create a text file with PDF paths, one per line:

```text
/path/to/document1.pdf
/path/to/document2.pdf
/path/to/subfolder/document3.pdf
# Comments are supported (lines starting with #)
/path/to/document4.pdf
```

## Key Changes from Original

1. **Input Method**:
   - Original: Single PDF path in config
   - New: Text file with multiple PDF paths

2. **Output Structure**:
   - Original: All outputs in one directory
   - New: Separate folder per PDF

3. **Image Saving**:
   - Original: Saves all detected images
   - New: Only saves images with captions

4. **Processing**:
   - Original: Sequential processing
   - New: Concurrent processing (max 10 PDFs)

## Performance Considerations

- **GPU Memory**: The max concurrency for image processing within each PDF is controlled by `MAX_CONCURRENCY` in config.py
- **System Memory**: Processing multiple PDFs simultaneously requires sufficient RAM
- **Disk I/O**: Ensure fast storage for optimal performance with concurrent writes

## Monitoring Progress

The script provides detailed progress information:
- Pre-processing progress for each PDF's images
- Overall PDF processing progress
- Summary statistics at completion

## Error Handling

- Invalid PDF paths are skipped with warnings
- PDFs that fail processing are logged but don't stop the batch
- Each PDF is processed independently to prevent cascading failures

## Example Workflow

1. Create your PDF paths file:
```bash
echo "/path/to/pdf1.pdf" > pdf_list.txt
echo "/path/to/pdf2.pdf" >> pdf_list.txt
```

2. Run the batch processor:
```bash
python run_dpsk_ocr_pdf_batch.py pdf_list.txt ./results
```

3. Check the results:
```bash
ls -la ./results/
# Each PDF will have its own folder with extracted images and captions
```

## Configuration

The script still uses the settings from `config.py` for:
- Model path
- Image processing parameters
- OCR prompt
- Other processing options

To adjust these settings, modify the `config.py` file as needed.