/* ==========================================================
   ANIMATIONS.JS
   IntersectionObserver scroll reveals + hero entrance
   ========================================================== */

export function initScrollReveal() {
  const reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  reveals.forEach((el) => observer.observe(el));
}

export function animateHeroEntrance() {
  const elements = document.querySelectorAll('.hero-enter');
  // Small delay to let the page paint first
  requestAnimationFrame(() => {
    elements.forEach((el) => el.classList.add('animate'));
  });
}

export function animateCounter(element, targetValue, duration = 1200) {
  const start = performance.now();
  const initial = 0;

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = initial + (targetValue - initial) * eased;
    element.textContent = current.toFixed(1) + '%';

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}
