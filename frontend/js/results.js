/* ==========================================================
   RESULTS.JS
   Verdict display, probability ring animation, detail toggle
   ========================================================== */

import { animateCounter } from './animations.js';

export function renderResults(data, imageUrl) {
  const section = document.getElementById('section-results');
  if (!section) return;

  // Verdict
  const verdictEl = document.getElementById('results-verdict');
  const assessmentEl = document.getElementById('results-assessment');

  const isFake = data.verdict === 'LIKELY FAKE';
  verdictEl.textContent = data.verdict;
  verdictEl.className = 'results__verdict ' + (isFake ? 'fake' : 'real');

  if (isFake) {
    assessmentEl.textContent =
      'The model estimates that this image is more likely to have been manipulated or AI-generated. This is a probabilistic assessment, not a definitive determination.';
  } else {
    assessmentEl.textContent =
      'The model estimates that this image is more likely to be authentic. This is a probabilistic assessment, not a definitive determination.';
  }

  // Visual Image
  const resultImgEl = document.getElementById('results-image');
  if (resultImgEl && imageUrl) {
    resultImgEl.src = imageUrl;
  }

  // Probability rendering
  const probsContainer = document.querySelector('.results__probabilities');
  const scaleContainer = document.querySelector('.results__prob-scale-container');

  if (typeof data.fake_probability === 'number' && typeof data.real_probability === 'number') {
    if (probsContainer) probsContainer.style.display = '';
    if (scaleContainer) scaleContainer.style.display = '';

    // Probability rings
    renderProbRing(
      'ring-fake',
      data.fake_probability * 100,
      isFake ? '--color-fake' : '--text-secondary'
    );
    renderProbRing(
      'ring-real',
      data.real_probability * 100,
      !isFake ? '--color-real' : '--text-secondary'
    );

    // Counter animations
    const fakePctEl = document.getElementById('fake-pct');
    const realPctEl = document.getElementById('real-pct');
    if (fakePctEl) animateCounter(fakePctEl, data.fake_probability * 100);
    if (realPctEl) animateCounter(realPctEl, data.real_probability * 100);

    // Probability scale marker
    const markerEl = document.getElementById('prob-scale-marker');
    if (markerEl) {
      const realProbPct = data.real_probability * 100;
      // Animate the marker after a short delay
      setTimeout(() => {
        markerEl.style.left = `${realProbPct}%`;
      }, 100);
    }
  } else {
    // Hide probabilities for endpoints without ML probability models (e.g., Text)
    if (probsContainer) probsContainer.style.display = 'none';
    if (scaleContainer) scaleContainer.style.display = 'none';
  }

  // Meta values
  document.getElementById('meta-model').textContent = data.model_name;
  const metaType = document.getElementById('meta-type');
  if (metaType) metaType.textContent = 'Image Forensics';
  document.getElementById('meta-threshold').textContent =
    (data.threshold * 100).toFixed(0) + '%';

  // Forensic details
  const detailInput = document.getElementById('detail-input');
  if (detailInput) detailInput.textContent = '224 × 224';
  
  // AI Generation Detection
  if (data.ai_generated) {
    const ai = data.ai_generated;
    const aiVerdictEl = document.getElementById('ai-verdict');
    const aiProbEl = document.getElementById('ai-prob');
    const aiStatusEl = document.getElementById('ai-status');

    if (ai.available) {
      aiVerdictEl.textContent = ai.assessment;
      aiVerdictEl.className = 'results__verdict ' + (ai.assessment === 'LIKELY AI-GENERATED' ? 'fake' : 'real');
      aiProbEl.textContent = (ai.ai_probability * 100).toFixed(1) + '%';
      aiStatusEl.textContent = 'Active (EfficientNet-B0 fine-tuned)';
    } else {
      aiVerdictEl.textContent = 'NOT AVAILABLE';
      aiVerdictEl.className = 'results__verdict';
      aiProbEl.textContent = '—';
      aiStatusEl.textContent = ai.error || 'Model checkpoint not found';
    }
  }
  
  // Image Integrity & Encoding
  if (data.integrity && data.integrity.available) {
    const integ = data.integrity;
    
    document.getElementById('integrity-format').textContent = integ.detected_format;
    document.getElementById('integrity-dims').textContent = `${integ.width} × ${integ.height}`;
    document.getElementById('integrity-color-mode').textContent = integ.color_mode;
    document.getElementById('integrity-filesize').textContent = Math.round(integ.file_size_bytes / 1024) + ' KB';
    document.getElementById('integrity-icc').textContent = integ.icc_profile_present ? 'Present' : 'Not detected';
    
    if (integ.analysis_note) {
      const noteEl = document.getElementById('integrity-note');
      noteEl.textContent = integ.analysis_note;
      noteEl.style.display = 'block';
    } else {
      document.getElementById('integrity-note').style.display = 'none';
    }
    
    if (integ.jpeg && integ.jpeg.available) {
      document.getElementById('integrity-jpeg-props').style.display = 'block';
      document.getElementById('integrity-jpeg-encoding').textContent = integ.jpeg.progressive ? 'Progressive' : 'Baseline';
      document.getElementById('integrity-subsampling').textContent = integ.jpeg.subsampling || 'N/A';
      document.getElementById('integrity-quantization').textContent = integ.jpeg.quantization_tables ? 'Detected' : 'Not available';
    } else {
      document.getElementById('integrity-jpeg-props').style.display = 'none';
    }
  }

  // Metadata Forensics
  if (data.metadata) {
    const meta = data.metadata;
    
    // EXIF
    document.getElementById('meta-exif').textContent = meta.exif_present ? 'AVAILABLE' : 'NOT PRESENT';
    
    // Camera
    let cameraStr = [];
    if (meta.camera_make) cameraStr.push(meta.camera_make);
    if (meta.camera_model) cameraStr.push(meta.camera_model);
    document.getElementById('meta-camera').textContent = cameraStr.length > 0 ? cameraStr.join(' ') : '—';
    
    // Software
    document.getElementById('meta-software').textContent = meta.software ? `Software metadata: ${meta.software}` : '—';
    
    // Capture Time
    document.getElementById('meta-capture-time').textContent = meta.datetime_original || meta.datetime || '—';
    
    // GPS
    document.getElementById('meta-gps').textContent = meta.gps_present ? 'PRESENT' : 'NOT PRESENT';
    
    // Dimensions
    if (meta.width && meta.height) {
      document.getElementById('meta-dims').textContent = `${meta.width} × ${meta.height}`;
    } else {
      document.getElementById('meta-dims').textContent = '—';
    }
  }
  
  // Error Level Analysis
  if (data.ela && data.ela.available) {
    const ela = data.ela;
    
    document.getElementById('ela-format').textContent = ela.format;
    document.getElementById('ela-quality').textContent = 'JPEG ' + ela.quality;
    document.getElementById('ela-mean').textContent = ela.mean_error;
    document.getElementById('ela-max').textContent = ela.max_error;
    
    if (ela.analysis_note) {
      const noteEl = document.getElementById('ela-note');
      noteEl.textContent = ela.analysis_note;
      noteEl.style.display = 'block';
    } else {
      document.getElementById('ela-note').style.display = 'none';
    }
    
    if (ela.visualization) {
      const imgEl = document.getElementById('ela-image');
      imgEl.src = ela.visualization;
      imgEl.style.display = 'block';
    }
  }

  // Visual Forensic Signals (Artifacts)
  if (data.artifacts && data.artifacts.available) {
    const arts = data.artifacts;
    const signalsSection = document.getElementById('forensic-signals-section');
    signalsSection.style.display = 'block';

    // Noise
    if (arts.noise_residual && arts.noise_residual.available) {
      document.getElementById('card-noise').style.display = 'block';
      document.getElementById('img-noise').src = arts.noise_residual.visualization;
      document.getElementById('val-noise-mean').textContent = arts.noise_residual.mean_absolute_residual.toFixed(2);
      document.getElementById('val-noise-max').textContent = arts.noise_residual.max_absolute_residual.toFixed(2);
    } else {
      document.getElementById('card-noise').style.display = 'none';
    }

    // Local Variation
    if (arts.local_variation && arts.local_variation.available) {
      document.getElementById('card-variation').style.display = 'block';
      document.getElementById('img-variation').src = arts.local_variation.visualization;
      document.getElementById('val-var-mean').textContent = arts.local_variation.mean.toFixed(2);
      document.getElementById('val-var-max').textContent = arts.local_variation.max.toFixed(2);
    } else {
      document.getElementById('card-variation').style.display = 'none';
    }

    // Edges
    if (arts.edge_analysis && arts.edge_analysis.available) {
      document.getElementById('card-edges').style.display = 'block';
      document.getElementById('img-edges').src = arts.edge_analysis.visualization;
      document.getElementById('val-edge-density').textContent = (arts.edge_analysis.edge_density * 100).toFixed(1) + '%';
    } else {
      document.getElementById('card-edges').style.display = 'none';
    }

    // JPEG Blocks
    if (arts.jpeg_block_analysis && arts.jpeg_block_analysis.available) {
      document.getElementById('card-jpeg').style.display = 'block';
      document.getElementById('img-jpeg').src = arts.jpeg_block_analysis.visualization;
    } else {
      document.getElementById('card-jpeg').style.display = 'none';
    }

    // FFT
    if (arts.frequency_analysis && arts.frequency_analysis.available) {
      document.getElementById('card-fft').style.display = 'block';
      document.getElementById('img-fft').src = arts.frequency_analysis.visualization;
    } else {
      document.getElementById('card-fft').style.display = 'none';
    }
  } else {
    const signalsSection = document.getElementById('forensic-signals-section');
    if (signalsSection) signalsSection.style.display = 'none';
  }
}

function renderProbRing(containerId, percentage, colorVar) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const svg = container.querySelector('.prob-ring__svg');
  const fillCircle = container.querySelector('.prob-ring__fill');
  if (!svg || !fillCircle) return;

  const radius = 105;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  const color = getComputedStyle(document.documentElement)
    .getPropertyValue(colorVar)
    .trim();

  fillCircle.style.stroke = color;
  fillCircle.setAttribute('stroke-dasharray', circumference);
  fillCircle.setAttribute('r', radius);

  // Animate from full offset to target
  fillCircle.style.strokeDashoffset = circumference;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fillCircle.style.strokeDashoffset = offset;
    });
  });
}

export function initDetailsToggle() {
  // Details panels have been flattened in the UI.
}
