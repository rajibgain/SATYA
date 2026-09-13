/* ==========================================================
   SCANNER.JS
   Scanning animation sequence during analysis
   ========================================================== */

const STAGES = [
  '01 IMAGE INGESTION',
  '02 IMAGE PREPROCESSING',
  '03 FEATURE EXTRACTION',
  '04 EFFICIENTNET-B0',
  '05 PROBABILITY ASSESSMENT',
];

export function runScanSequence(onComplete) {
  const stageEl = document.getElementById('scanner-stage');
  const progressFill = document.getElementById('scanner-progress-fill');
  const percentEl = document.getElementById('scanner-percent');

  if (!stageEl || !progressFill || !percentEl) return;

  let currentStage = 0;
  const totalStages = STAGES.length;
  const stageInterval = 500; // ms per stage

  function updateStage() {
    if (currentStage >= totalStages) {
      if (onComplete) onComplete();
      return;
    }

    const progress = ((currentStage + 1) / totalStages) * 100;

    stageEl.textContent = STAGES[currentStage];
    progressFill.style.width = progress + '%';
    percentEl.textContent = Math.round(progress) + '%';

    currentStage++;
    setTimeout(updateStage, stageInterval);
  }

  // Reset state
  stageEl.textContent = '';
  progressFill.style.width = '0%';
  percentEl.textContent = '0%';

  // Start with a small delay for visual polish
  setTimeout(updateStage, 200);
}
