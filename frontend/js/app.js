/* ==========================================================
   APP.JS
   SPA router, navigation, initialization
   ========================================================== */

import { initScrollReveal, animateHeroEntrance } from './animations.js';
import { initAnalyzer, resetAnalyzer, setAnalyzerMode } from './analyzer.js';
import { initDetailsToggle } from './results.js';

const SECTIONS = ['home', 'analyze', 'results', 'methodology', 'limitations'];

// ── Navigation ──
function navigate(sectionId) {
  SECTIONS.forEach((id) => {
    const el = document.getElementById('section-' + id);
    if (el) {
      el.classList.toggle('active', id === sectionId);
    }
  });

  // Update nav link active state
  document.querySelectorAll('.nav__link').forEach((link) => {
    const target = link.dataset.section;
    link.classList.toggle('active', target === sectionId);
  });

  // Update hash without triggering scroll
  history.pushState(null, '', '#' + sectionId);

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'instant' });

  // Re-init scroll reveals for new section
  requestAnimationFrame(() => initScrollReveal());
}

// ── Hash routing ──
function handleHash() {
  const hash = location.hash.slice(1) || 'home';
  if (SECTIONS.includes(hash)) {
    navigate(hash);
  } else {
    navigate('home');
  }
}

// ── Mobile nav toggle ──
function initMobileNav() {
  const toggle = document.getElementById('nav-toggle');
  const links = document.getElementById('nav-links');

  if (toggle && links) {
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
    });

    // Close menu on link click
    links.querySelectorAll('.nav__link').forEach((link) => {
      link.addEventListener('click', () => {
        links.classList.remove('open');
      });
    });
  }
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  // Wire up navigation links
  document.querySelectorAll('[data-section]').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      
      const target = el.dataset.section;
      if (target) navigate(target);
      
      const mode = el.dataset.mode;
      if (mode) setAnalyzerMode(mode);
    });
  });

  // "Analyze Another" button
  document.getElementById('btn-analyze-another')?.addEventListener('click', () => {
    resetAnalyzer();
    navigate('analyze');
  });

  // Init modules
  initMobileNav();
  initAnalyzer(navigate);
  initDetailsToggle();

  // Handle initial hash
  handleHash();
  window.addEventListener('hashchange', handleHash);

  // Hero entrance animation (only if on home)
  if (!location.hash || location.hash === '#home') {
    animateHeroEntrance();
  }

  // Init scroll reveals
  initScrollReveal();
});
