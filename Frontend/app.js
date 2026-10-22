"use strict";

const API_BASE_URL = "http://127.0.0.1:8000";
const API_URL = `${API_BASE_URL}/predict`;
const LABELS = ["negative", "neutral", "positive"];

const form = document.querySelector("#review-form");
const reviewInput = document.querySelector("#review");
const summaryInput = document.querySelector("#summary");
const characterCount = document.querySelector("#character-count");
const analyzeButton = document.querySelector("#analyze-button");
const formMessage = document.querySelector("#form-message");
const emptyState = document.querySelector("#empty-state");
const prediction = document.querySelector("#prediction");
const sentimentPill = document.querySelector("#sentiment-pill");
const sentimentValue = document.querySelector("#sentiment-value");
const confidenceValue = document.querySelector("#confidence-value");
const confidenceBar = document.querySelector("#confidence-bar");
const classScores = document.querySelector("#class-scores");
const apiStatus = document.querySelector("#api-status");
const apiStatusText = document.querySelector("#api-status-text");
const explanationEmpty = document.querySelector("#explanation-empty");
const explanationContent = document.querySelector("#explanation-content");
const comparisonLabel = document.querySelector("#comparison-label");
const supportingSignals = document.querySelector("#supporting-signals");
const opposingSignals = document.querySelector("#opposing-signals");

checkApiStatus();

reviewInput.addEventListener("input", () => {
  characterCount.textContent = `${reviewInput.value.length} / 5000`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formMessage.hidden = true;

  const review = reviewInput.value.trim();
  const summary = summaryInput.value.trim();
  if (!review && !summary) {
    showError("Enter a review or summary before analyzing.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review, summary }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "The API could not analyze this review.");
    }
    showResult(result);
  } catch (error) {
    showError(
      error instanceof TypeError
        ? "Could not reach the API. Check that the backend is running in the other terminal."
        : error.message,
    );
  } finally {
    setLoading(false);
  }
});

function showResult(result) {
  emptyState.hidden = true;
  prediction.hidden = false;
  prediction.classList.remove("result-enter");
  void prediction.offsetWidth;
  prediction.classList.add("result-enter");
  sentimentValue.textContent = result.sentiment;
  sentimentPill.className = `sentiment-pill ${result.sentiment}`;

  const confidencePercent = Number(result.confidence_estimate) * 100;
  confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
  confidenceBar.style.width = `${Math.min(100, Math.max(0, confidencePercent))}%`;

  classScores.replaceChildren();
  for (const label of LABELS) {
    const score = Number(result.class_probability_estimates[label] || 0);
    const row = document.createElement("div");
    row.className = "score-row";

    const labelElement = document.createElement("span");
    labelElement.className = "score-label";
    labelElement.textContent = label;

    const track = document.createElement("div");
    track.className = "score-track";
    track.setAttribute("aria-label", `${label} probability estimate`);
    const bar = document.createElement("span");
    const scorePercent = Math.min(100, Math.max(0, score * 100));
    bar.style.width = "0%";
    track.append(bar);

    const value = document.createElement("span");
    value.className = "score-value";
    value.textContent = `${(score * 100).toFixed(1)}%`;

    row.append(labelElement, track, value);
    classScores.append(row);
    requestAnimationFrame(() => {
      bar.style.width = `${scorePercent}%`;
    });
  }

  showExplanation(result.explanation);
}

function showExplanation(explanation) {
  explanationEmpty.hidden = true;
  explanationContent.hidden = false;
  const competingSentiment = explanation?.compared_with || "another class";
  comparisonLabel.textContent = `The model compared ${resultLabel(sentimentValue.textContent)} with ${competingSentiment}.`;
  renderSignals(supportingSignals, explanation?.supporting_terms || [], "supporting");
  renderSignals(opposingSignals, explanation?.opposing_terms || [], "opposing");
}

function renderSignals(container, terms, direction) {
  container.replaceChildren();
  if (terms.length === 0) {
    const empty = document.createElement("span");
    empty.className = "no-signals";
    empty.textContent = "No clear terms from the learned vocabulary.";
    container.append(empty);
    return;
  }

  for (const item of terms) {
    const signal = document.createElement("span");
    signal.className = `signal-chip ${direction}`;
    signal.textContent = item.term;
    container.append(signal);
  }
}

function resultLabel(label) {
  return label ? label[0].toUpperCase() + label.slice(1) : "the selected sentiment";
}

function showError(message) {
  formMessage.textContent = message;
  formMessage.hidden = false;
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.firstElementChild.textContent = isLoading ? "ANALYZING..." : "ANALYZE REVIEW";
}

async function checkApiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("API health check failed");
    apiStatus.classList.add("online");
    apiStatus.classList.remove("offline");
    apiStatusText.textContent = "API connected";
  } catch {
    apiStatus.classList.add("offline");
    apiStatus.classList.remove("online");
    apiStatusText.textContent = "API offline";
  }
}
