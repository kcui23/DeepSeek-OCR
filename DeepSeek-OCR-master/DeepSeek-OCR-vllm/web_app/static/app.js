// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeBtn = document.getElementById('removeBtn');
const promptInput = document.getElementById('promptInput');
const runBtn = document.getElementById('runBtn');
const runBtnText = document.getElementById('runBtnText');
const spinner = document.getElementById('spinner');

// Parameters
const baseSize = document.getElementById('baseSize');
const baseSizeValue = document.getElementById('baseSizeValue');
const imageSize = document.getElementById('imageSize');
const imageSizeValue = document.getElementById('imageSizeValue');
const cropMode = document.getElementById('cropMode');
const minCrops = document.getElementById('minCrops');
const minCropsValue = document.getElementById('minCropsValue');
const maxCrops = document.getElementById('maxCrops');
const maxCropsValue = document.getElementById('maxCropsValue');

// Results
const emptyState = document.getElementById('emptyState');
const resultsContent = document.getElementById('resultsContent');
const tabBtns = document.querySelectorAll('.tab-btn');
const visualizedImage = document.getElementById('visualizedImage');
const noVisualMsg = document.getElementById('noVisualMsg');
const processedText = document.getElementById('processedText');
const rawText = document.getElementById('rawText');

// State
let uploadedFilePath = null;
let currentOutputId = null;

// Initialize
init();

function init() {
    // Upload area click
    uploadPlaceholder.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('dragleave', handleDragLeave);

    // Remove image
    removeBtn.addEventListener('click', removeImage);

    // Parameter sliders
    baseSize.addEventListener('input', (e) => {
        baseSizeValue.textContent = e.target.value;
    });

    imageSize.addEventListener('input', (e) => {
        imageSizeValue.textContent = e.target.value;
    });

    minCrops.addEventListener('input', (e) => {
        minCropsValue.textContent = e.target.value;
    });

    maxCrops.addEventListener('input', (e) => {
        maxCropsValue.textContent = e.target.value;
    });

    // Run button
    runBtn.addEventListener('click', runInference);

    // Tab buttons
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('请上传图片文件');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Upload to server
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            uploadedFilePath = data.path;
            runBtn.disabled = false;
        } else {
            alert('上传失败：' + data.detail);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('上传失败：' + error.message);
    }
}

function removeImage() {
    previewImg.src = '';
    uploadPlaceholder.style.display = 'flex';
    imagePreview.style.display = 'none';
    fileInput.value = '';
    uploadedFilePath = null;
    runBtn.disabled = true;
}

async function runInference() {
    if (!uploadedFilePath) {
        alert('请先上传图片');
        return;
    }

    // Disable button and show loading
    runBtn.disabled = true;
    runBtnText.textContent = '处理中...';
    spinner.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('image_path', uploadedFilePath);
        formData.append('prompt', promptInput.value);
        formData.append('base_size', baseSize.value);
        formData.append('image_size', imageSize.value);
        formData.append('crop_mode', cropMode.checked);
        formData.append('min_crops', minCrops.value);
        formData.append('max_crops', maxCrops.value);

        const response = await fetch('/inference', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Inference failed');
        }

        const data = await response.json();

        if (data.success) {
            currentOutputId = data.output_id;
            displayResults(data);
        } else {
            alert('推理失败');
        }
    } catch (error) {
        console.error('Inference error:', error);
        alert('推理失败：' + error.message);
    } finally {
        // Re-enable button
        runBtn.disabled = false;
        runBtnText.textContent = '运行 OCR';
        spinner.style.display = 'none';
    }
}

function displayResults(data) {
    // Hide empty state
    emptyState.style.display = 'none';
    resultsContent.style.display = 'block';

    // Display visualized image
    if (data.has_visualized_image) {
        visualizedImage.src = `/output/${data.output_id}/result_with_boxes.jpg`;
        visualizedImage.style.display = 'block';
        noVisualMsg.style.display = 'none';
    } else {
        visualizedImage.style.display = 'none';
        noVisualMsg.style.display = 'block';
    }

    // Display processed result
    processedText.textContent = data.processed_result;

    // Display raw result
    rawText.textContent = data.raw_result;

    // Switch to visualized tab
    switchTab('visualized');
}

function switchTab(tabName) {
    // Update tab buttons
    tabBtns.forEach(btn => {
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });

    const activeTab = document.getElementById(tabName + 'Tab');
    if (activeTab) {
        activeTab.classList.add('active');
    }
}
