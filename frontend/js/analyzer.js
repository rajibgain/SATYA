/* ==========================================================
   ANALYZER.JS
   Upload handling, preview, API call, scanning orchestration
   ========================================================== */

import { runScanSequence } from './scanner.js';
import { renderResults } from './results.js';

const API_URLS = {
  image: '/api/analyze',
  video: '/api/analyze/video',
  text: '/api/analyze/text'
};

const ALLOWED_TYPES_MAP = {
  image: ['image/jpeg', 'image/png', 'image/jpg'],
  video: ['video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska'],
  text: ['text/plain']
};

const MAX_SIZE_MB = 20;

let currentMode = 'image';
let selectedFile = null;

export function setAnalyzerMode(mode) {
  if (!API_URLS[mode]) return;
  currentMode = mode;
  resetUpload();

  const title = document.getElementById('upload-title');
  const formats = document.getElementById('upload-formats');
  const fileInput = document.getElementById('file-input');

  if (mode === 'image') {
    title.textContent = 'Drop image for forensic analysis';
    formats.textContent = 'Supported: JPG / JPEG / PNG';
    fileInput.accept = 'image/jpeg,image/png,image/jpg';
  } else if (mode === 'video') {
    title.textContent = 'Drop video for forensic analysis';
    formats.textContent = 'Supported: MP4 / AVI / MOV / MKV';
    fileInput.accept = 'video/mp4,video/x-msvideo,video/quicktime,video/x-matroska';
  } else if (mode === 'text') {
    title.textContent = 'Drop text file for claim verification';
    formats.textContent = 'Supported: TXT';
    fileInput.accept = 'text/plain';
  }
}

export function initAnalyzer(navigateFn) {
  const zone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const preview = document.getElementById('preview');
  const scanner = document.getElementById('scanner');
  const analyzeBtn = document.getElementById('btn-analyze');
  const removeBtn = document.getElementById('btn-remove');

  if (!zone) return;

  // Click to upload
  zone.addEventListener('click', () => fileInput.click());

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  // Drag and drop
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('dragover');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  // Analyze button
  analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    startAnalysis(navigateFn);
  });

  // Remove / replace
  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });
}

function handleFile(file) {
  // Validate type
  const allowed = ALLOWED_TYPES_MAP[currentMode];
  if (!allowed.includes(file.type)) {
    alert(`Unsupported file type for ${currentMode} mode.`);
    return;
  }

  // Validate size
  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    alert(`File too large. Maximum size is ${MAX_SIZE_MB}MB.`);
    return;
  }

  selectedFile = file;
  showPreview(file);
}

function showPreview(file) {
  const zone = document.getElementById('upload-zone');
  const preview = document.getElementById('preview');
  const previewImg = document.getElementById('preview-image');
  const metaName = document.getElementById('meta-filename');
  const metaDims = document.getElementById('meta-dimensions');
  const metaSize = document.getElementById('meta-filesize');
  const metaType = document.getElementById('meta-filetype');

  // Hide upload zone, show preview
  zone.style.display = 'none';
  preview.classList.add('active');

  // Load image for preview and dimension reading
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;

    // For non-images, we don't display a real preview image
    if (currentMode !== 'image') {
      metaDims.textContent = 'N/A';
      previewImg.src = ''; 
    } else {
      // Read dimensions once loaded
      const img = new Image();
      img.onload = () => {
        metaDims.textContent = img.width + ' × ' + img.height;
      };
      img.src = e.target.result;
    }
  };
  reader.readAsDataURL(file);

  // Metadata
  metaName.textContent = file.name;
  metaSize.textContent = formatFileSize(file.size);
  metaType.textContent = file.type.split('/')[1].toUpperCase();
}

function resetUpload() {
  selectedFile = null;

  const zone = document.getElementById('upload-zone');
  const preview = document.getElementById('preview');
  const scanner = document.getElementById('scanner');
  const fileInput = document.getElementById('file-input');

  zone.style.display = '';
  preview.classList.remove('active');
  scanner.classList.remove('active');
  fileInput.value = '';
}

async function startAnalysis(navigateFn) {
  const preview = document.getElementById('preview');
  const scanner = document.getElementById('scanner');
  const scannerImg = document.getElementById('scanner-image');

  // Switch to scanner view
  preview.classList.remove('active');
  scanner.classList.add('active');

  // Show the image in scanner
  const reader = new FileReader();
  reader.onload = (e) => {
    scannerImg.src = e.target.result;
  };
  reader.readAsDataURL(selectedFile);

  // Start scanning animation and API call in parallel
  const formData = new FormData();
  formData.append(currentMode, selectedFile);

  let apiResult = null;
  let scanComplete = false;

  // API call
  const apiUrl = API_URLS[currentMode];
  const apiPromise = fetch(apiUrl, {
    method: 'POST',
    body: formData,
  })
    .then((res) => {
      if (!res.ok) throw new Error('Analysis failed');
      return res.json();
    })
    .then((data) => {
      apiResult = data;
    })
    .catch((err) => {
      console.error('API Error:', err);
      alert('Analysis failed. Make sure the server is running.');
      resetUpload();
    });

  // Scanning animation
  runScanSequence(() => {
    scanComplete = true;
    // If API already returned, show results immediately
    if (apiResult) {
      showResults(apiResult, navigateFn);
    }
  });

  // If API finishes before animation, wait for animation
  await apiPromise;
  if (apiResult && scanComplete) {
    showResults(apiResult, navigateFn);
  } else if (apiResult && !scanComplete) {
    // API finished first — wait handled by scanSequence callback above
  }
}

function showResults(data, navigateFn) {
  const imageUrl = document.getElementById('preview-image').src;
  renderResults(data, imageUrl);
  navigateFn('results');
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export function resetAnalyzer() {
  resetUpload();
}
