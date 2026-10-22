"use strict";

// Decorative, lightweight depth effect behind the dashboard.
const canvas = document.querySelector("#network-background");
const context = canvas.getContext("2d");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let width = 0;
let height = 0;
let pixelRatio = 1;
let points = [];
let lastFrameTime = 0;
let animationFrameId = null;

function createPoints() {
  const pointCount = Math.max(30, Math.min(64, Math.round(width / 23)));
  points = Array.from({ length: pointCount }, () => {
    const depth = 0.25 + Math.random() * 0.75;
    const drift = (1.2 - depth) * 0.19;
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      depth,
      radius: 0.9 + (1 - depth) * 1.45,
      opacity: 0.11 + (1 - depth) * 0.22,
      dx: (Math.random() - 0.5) * drift,
      dy: (Math.random() - 0.5) * drift,
    };
  });
}

function resizeCanvas() {
  width = window.innerWidth;
  height = window.innerHeight;
  pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  createPoints();
  drawFrame();
}

function drawFrame() {
  context.clearRect(0, 0, width, height);

  for (let first = 0; first < points.length; first += 1) {
    const point = points[first];
    for (let second = first + 1; second < points.length; second += 1) {
      const other = points[second];
      const distance = Math.hypot(point.x - other.x, point.y - other.y);
      const depth = (point.depth + other.depth) / 2;
      const connectionRange = 94 + (1 - depth) * 43;
      if (distance >= connectionRange) continue;

      const fade = 1 - distance / connectionRange;
      context.beginPath();
      context.moveTo(point.x, point.y);
      context.lineTo(other.x, other.y);
      context.strokeStyle = `rgba(22, 200, 233, ${fade * (0.08 + (1 - depth) * 0.24)})`;
      context.lineWidth = 0.6 + (1 - depth) * 0.5;
      context.stroke();
    }
  }

  for (const point of points) {
    context.beginPath();
    context.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
    context.fillStyle = `rgba(92, 224, 242, ${point.opacity})`;
    context.shadowColor = `rgba(22, 200, 233, ${point.opacity * 0.7})`;
    context.shadowBlur = 5 + (1 - point.depth) * 5;
    context.fill();
  }
  context.shadowBlur = 0;
}

function animate(time) {
  if (reducedMotion.matches) {
    animationFrameId = null;
    return;
  }
  animationFrameId = window.requestAnimationFrame(animate);
  if (time - lastFrameTime < 40) return;

  const elapsed = lastFrameTime ? Math.min(time - lastFrameTime, 48) : 16;
  lastFrameTime = time;
  for (const point of points) {
    point.x += point.dx * elapsed;
    point.y += point.dy * elapsed;
    if (point.x < -8) point.x = width + 8;
    if (point.x > width + 8) point.x = -8;
    if (point.y < -8) point.y = height + 8;
    if (point.y > height + 8) point.y = -8;
  }
  drawFrame();
}

function handleMotionPreference() {
  if (reducedMotion.matches) {
    if (animationFrameId !== null) window.cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
    drawFrame();
  } else if (animationFrameId === null) {
    lastFrameTime = 0;
    animationFrameId = window.requestAnimationFrame(animate);
  }
}

window.addEventListener("resize", resizeCanvas, { passive: true });
reducedMotion.addEventListener?.("change", handleMotionPreference);
resizeCanvas();
if (!reducedMotion.matches) animationFrameId = window.requestAnimationFrame(animate);
