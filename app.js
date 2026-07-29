"use strict";

const hasDOM = typeof document !== "undefined";
const $ = (id) => (hasDOM ? document.getElementById(id) : null);
const svgNS = "http://www.w3.org/2000/svg";
const SERIES = [
  { key: "raw", label: "Raw", color: "var(--ink)" },
  { key: "temperature", label: "Temperature", color: "var(--primary)" },
  { key: "isotonic", label: "Isotonic", color: "var(--secondary)" },
];

let data;
let currentAnalysis;
const cache = {};

function clampProbability(value) {
  return Math.min(1 - 1e-12, Math.max(1e-12, value));
}

function binaryNll(yTrue, probability) {
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const p = clampProbability(probability[i]);
    total -= yTrue[i] * Math.log(p) + (1 - yTrue[i]) * Math.log(1 - p);
  }
  return total / yTrue.length;
}

function temperatureScale(probability, temperature) {
  return probability.map((value) => {
    const p = clampProbability(value);
    const logit = Math.log(p / (1 - p)) / temperature;
    return 1 / (1 + Math.exp(-logit));
  });
}

function fitTemperature(probability, yTrue) {
  let left = Math.log(0.05);
  let right = Math.log(20);
  for (let i = 0; i < 80; i += 1) {
    const third = (right - left) / 3;
    const a = left + third;
    const b = right - third;
    const lossA = binaryNll(yTrue, temperatureScale(probability, Math.exp(a)));
    const lossB = binaryNll(yTrue, temperatureScale(probability, Math.exp(b)));
    if (lossA < lossB) right = b;
    else left = a;
  }
  return Math.exp((left + right) / 2);
}

function fitIsotonic(scores, yTrue) {
  const pairs = scores.map((score, i) => ({ score, positives: yTrue[i], count: 1 }));
  pairs.sort((a, b) => a.score - b.score);

  const grouped = [];
  for (const pair of pairs) {
    const last = grouped[grouped.length - 1];
    if (last && last.score === pair.score) {
      last.positives += pair.positives;
      last.count += 1;
    } else {
      grouped.push({ ...pair });
    }
  }

  const blocks = [];
  for (const group of grouped) {
    blocks.push({
      upper: group.score,
      positives: group.positives,
      count: group.count,
      value: group.positives / group.count,
    });
    while (blocks.length > 1 && blocks.at(-2).value > blocks.at(-1).value) {
      const right = blocks.pop();
      const left = blocks.pop();
      const count = left.count + right.count;
      const positives = left.positives + right.positives;
      blocks.push({
        upper: right.upper,
        positives,
        count,
        value: positives / count,
      });
    }
  }
  return blocks;
}

function applyIsotonic(probability, blocks) {
  return probability.map((value) => {
    let left = 0;
    let right = blocks.length - 1;
    while (left < right) {
      const middle = Math.floor((left + right) / 2);
      if (value <= blocks[middle].upper) right = middle;
      else left = middle + 1;
    }
    return blocks[left].value;
  });
}

function calibrationBins(yTrue, probability, binCount, scheme) {
  const bins = Array.from({ length: binCount }, () => ({ count: 0, sumP: 0, sumY: 0 }));
  if (scheme === "quantile") {
    const order = probability.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
    order.forEach((item, rank) => {
      const bin = bins[Math.min(binCount - 1, Math.floor((rank * binCount) / order.length))];
      bin.count += 1;
      bin.sumP += item.value;
      bin.sumY += yTrue[item.index];
    });
  } else {
    probability.forEach((value, index) => {
      const bin = bins[Math.min(binCount - 1, Math.floor(value * binCount))];
      bin.count += 1;
      bin.sumP += value;
      bin.sumY += yTrue[index];
    });
  }
  return bins.map((bin, index) => ({
    index: index + 1,
    count: bin.count,
    meanP: bin.count ? bin.sumP / bin.count : null,
    observed: bin.count ? bin.sumY / bin.count : null,
  }));
}

function calibrationError(yTrue, probability, binCount, scheme) {
  const bins = calibrationBins(yTrue, probability, binCount, scheme);
  let ece = 0;
  let mce = 0;
  for (const bin of bins) {
    if (!bin.count) continue;
    const gap = Math.abs(bin.meanP - bin.observed);
    ece += (bin.count / yTrue.length) * gap;
    mce = Math.max(mce, gap);
  }
  return { ece, mce, bins };
}

function brierScore(yTrue, probability) {
  return probability.reduce((sum, value, index) => sum + (value - yTrue[index]) ** 2, 0) / yTrue.length;
}

function murphyDecomposition(yTrue, probability) {
  const groups = new Map();
  probability.forEach((value, index) => {
    const group = groups.get(value) || { count: 0, positives: 0 };
    group.count += 1;
    group.positives += yTrue[index];
    groups.set(value, group);
  });

  const meanY = yTrue.reduce((sum, value) => sum + value, 0) / yTrue.length;
  let reliability = 0;
  let resolution = 0;
  for (const [probabilityValue, group] of groups) {
    const weight = group.count / yTrue.length;
    const observed = group.positives / group.count;
    reliability += weight * (probabilityValue - observed) ** 2;
    resolution += weight * (observed - meanY) ** 2;
  }
  const uncertainty = meanY * (1 - meanY);
  return {
    reliability,
    resolution,
    uncertainty,
    reconstructed: reliability - resolution + uncertainty,
  };
}

function aucByRank(yTrue, probability) {
  const pairs = probability.map((score, index) => ({ score, label: yTrue[index] }));
  pairs.sort((a, b) => a.score - b.score);
  let positiveRankSum = 0;
  let positives = 0;
  for (let start = 0; start < pairs.length;) {
    let end = start + 1;
    while (end < pairs.length && pairs[end].score === pairs[start].score) end += 1;
    const averageRank = (start + 1 + end) / 2;
    for (let i = start; i < end; i += 1) {
      if (pairs[i].label === 1) {
        positiveRankSum += averageRank;
        positives += 1;
      }
    }
    start = end;
  }
  const negatives = pairs.length - positives;
  return (positiveRankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

function metricsFor(yTrue, probability, binCount, scheme) {
  const calibration = calibrationError(yTrue, probability, binCount, scheme);
  const brier = brierScore(yTrue, probability);
  const murphy = murphyDecomposition(yTrue, probability);
  return {
    ece: calibration.ece,
    mce: calibration.mce,
    bins: calibration.bins,
    brier,
    reliability: murphy.reliability,
    resolution: murphy.resolution,
    uncertainty: murphy.uncertainty,
    reconstructed: murphy.reconstructed,
    reconstructionGap: Math.abs(brier - murphy.reconstructed),
    logLoss: binaryNll(yTrue, probability),
    auc: aucByRank(yTrue, probability),
  };
}

function computeAnalysis(modelKey) {
  if (cache[modelKey]) return cache[modelKey];
  const model = data.models[modelKey];
  const calibrationY = data.calibration.y_true;
  const testY = data.test.y_true;
  const temperature = fitTemperature(model.calibration_prob, calibrationY);
  const blocks = fitIsotonic(model.calibration_prob, calibrationY);
  const result = {
    model,
    temperature,
    blocks,
    calibrationNllRaw: binaryNll(calibrationY, model.calibration_prob),
    calibrationNllTemperature: binaryNll(
      calibrationY,
      temperatureScale(model.calibration_prob, temperature)
    ),
    probabilities: {
      raw: model.test_prob,
      temperature: temperatureScale(model.test_prob, temperature),
      isotonic: applyIsotonic(model.test_prob, blocks),
    },
    yTrue: testY,
  };
  cache[modelKey] = result;
  return result;
}

function svgElement(name, attributes = {}, text = "") {
  const element = document.createElementNS(svgNS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text) element.textContent = text;
  return element;
}

function renderChart(metricSets) {
  const svg = $("reliability-chart");
  svg.textContent = "";
  svg.append(
    svgElement("title", { id: "chart-title" }, "Reliability diagram with count histogram"),
    svgElement(
      "desc",
      { id: "chart-description" },
      "Raw, temperature-scaled, and isotonic observed rates against mean probabilities, with per-bin counts below."
    )
  );

  const left = 205;
  const top = 28;
  const plotSize = 350;
  const plotRight = left + plotSize;
  const plotBottom = top + plotSize;
  const histogramTop = 430;
  const histogramHeight = 72;
  const axisColor = "var(--hairline)";
  const mutedColor = "var(--muted)";

  for (let tick = 0; tick <= 5; tick += 1) {
    const value = tick / 5;
    const x = left + value * plotSize;
    const y = plotBottom - value * plotSize;
    svg.append(
      svgElement("line", { x1: x, y1: top, x2: x, y2: plotBottom, stroke: axisColor }),
      svgElement("line", { x1: left, y1: y, x2: plotRight, y2: y, stroke: axisColor }),
      svgElement("text", { x, y: plotBottom + 22, "text-anchor": "middle", fill: mutedColor, class: "svg-number" }, value.toFixed(1)),
      svgElement("text", { x: left - 12, y: y + 4, "text-anchor": "end", fill: mutedColor, class: "svg-number" }, value.toFixed(1))
    );
  }

  svg.append(
    svgElement("line", {
      x1: left,
      y1: plotBottom,
      x2: plotRight,
      y2: top,
      stroke: "var(--reference)",
      "stroke-dasharray": "5 5",
      "stroke-width": 1.5,
    }),
    svgElement("text", { x: (left + plotRight) / 2, y: 416, "text-anchor": "middle", fill: mutedColor, class: "svg-label" }, "Mean predicted probability"),
    svgElement("text", { x: 150, y: (top + plotBottom) / 2, transform: `rotate(-90 150 ${(top + plotBottom) / 2})`, "text-anchor": "middle", fill: mutedColor, class: "svg-label" }, "Observed positive rate"),
    svgElement("text", { x: left, y: histogramTop - 10, fill: mutedColor, class: "svg-label" }, "Cases per bin")
  );

  const maxCount = Math.max(
    1,
    ...SERIES.flatMap((series) => metricSets[series.key].bins.map((bin) => bin.count))
  );
  const binCount = metricSets.raw.bins.length;
  const groupWidth = plotSize / binCount;
  const barWidth = Math.max(1.5, (groupWidth - 4) / SERIES.length);

  SERIES.forEach((series, seriesIndex) => {
    const bins = metricSets[series.key].bins;
    const points = bins
      .filter((bin) => bin.count)
      .map((bin) => `${left + bin.meanP * plotSize},${plotBottom - bin.observed * plotSize}`)
      .join(" ");
    svg.appendChild(svgElement("polyline", {
      points,
      fill: "none",
      stroke: series.color,
      "stroke-width": 2.25,
      "stroke-linejoin": "round",
      "stroke-linecap": "round",
    }));

    bins.forEach((bin, index) => {
      if (bin.count) {
        svg.appendChild(svgElement("circle", {
          cx: left + bin.meanP * plotSize,
          cy: plotBottom - bin.observed * plotSize,
          r: 3.5,
          fill: "var(--card)",
          stroke: series.color,
          "stroke-width": 2,
        }));
      }
      const height = (bin.count / maxCount) * histogramHeight;
      svg.appendChild(svgElement("rect", {
        x: left + index * groupWidth + 2 + seriesIndex * barWidth,
        y: histogramTop + histogramHeight - height,
        width: barWidth,
        height,
        fill: series.color,
        opacity: 0.78,
      }));
    });
  });

  svg.append(
    svgElement("line", { x1: left, y1: histogramTop + histogramHeight, x2: plotRight, y2: histogramTop + histogramHeight, stroke: mutedColor }),
    svgElement("text", { x: left - 12, y: histogramTop + 4, "text-anchor": "end", fill: mutedColor, class: "svg-number" }, String(maxCount)),
    svgElement("text", { x: left - 12, y: histogramTop + histogramHeight + 4, "text-anchor": "end", fill: mutedColor, class: "svg-number" }, "0")
  );
}

function appendCells(row, values) {
  values.forEach((value) => {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  });
}

function renderReliabilityTable(metricSets) {
  const body = document.querySelector("#reliability-table tbody");
  body.textContent = "";
  SERIES.forEach((series) => {
    metricSets[series.key].bins.forEach((bin) => {
      const row = document.createElement("tr");
      const method = document.createElement("td");
      method.textContent = series.label;
      row.appendChild(method);
      appendCells(row, [
        String(bin.index),
        bin.count ? bin.meanP.toFixed(4) : "empty",
        bin.count ? bin.observed.toFixed(4) : "empty",
        String(bin.count),
      ]);
      body.appendChild(row);
    });
  });
}

function formatMetric(value, key) {
  if (key === "reconstructionGap") return value < 1e-12 ? value.toExponential(1) : value.toFixed(10);
  return value.toFixed(4);
}

function formatDelta(value) {
  if (Math.abs(value) < 5e-8) return "0.0000";
  return `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
}

function renderMetrics(metricSets) {
  const rows = [
    ["ECE, lower", "ece", -1],
    ["MCE, lower", "mce", -1],
    ["Brier score, lower", "brier", -1],
    ["Murphy reliability, lower", "reliability", -1],
    ["Murphy resolution, higher", "resolution", 1],
    ["Murphy uncertainty", "uncertainty", 0],
    ["REL - RES + UNC", "reconstructed", -1],
    ["Reconstruction gap", "reconstructionGap", -1],
    ["Log loss, lower", "logLoss", -1],
    ["Rank AUC, higher", "auc", 1],
  ];
  const body = document.querySelector("#metrics-table tbody");
  body.textContent = "";
  rows.forEach(([label, key, direction]) => {
    const raw = metricSets.raw[key];
    const temperature = metricSets.temperature[key];
    const isotonic = metricSets.isotonic[key];
    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = label;
    row.appendChild(heading);

    const values = [
      formatMetric(raw, key),
      formatMetric(temperature, key),
      formatDelta(temperature - raw),
      formatMetric(isotonic, key),
      formatDelta(isotonic - raw),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if ((index === 2 || index === 4) && direction) {
        const delta = index === 2 ? temperature - raw : isotonic - raw;
        if (Math.abs(delta) >= 5e-8) {
          cell.className = delta * direction > 0 ? "delta-better" : "delta-worse";
        }
      }
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function decisionStats(yTrue, probability, falsePositiveCost, falseNegativeCost) {
  const threshold = falsePositiveCost / (falsePositiveCost + falseNegativeCost);
  let predictedPositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let expectedCost = 0;
  probability.forEach((value, index) => {
    const predictsPositive = value >= threshold;
    if (predictsPositive) {
      predictedPositive += 1;
      falsePositive += yTrue[index] === 0 ? 1 : 0;
      expectedCost += (1 - value) * falsePositiveCost;
    } else {
      falseNegative += yTrue[index] === 1 ? 1 : 0;
      expectedCost += value * falseNegativeCost;
    }
  });
  return {
    predictedPositive,
    falsePositive,
    falseNegative,
    expectedCost: (expectedCost / yTrue.length) * 100,
    observedCost:
      ((falsePositive * falsePositiveCost + falseNegative * falseNegativeCost) / yTrue.length) * 100,
  };
}

function renderDecision() {
  if (!currentAnalysis) return;
  const falsePositiveCost = Math.max(0.1, Number($("false-positive-cost").value) || 0.1);
  const falseNegativeCost = Math.max(0.1, Number($("false-negative-cost").value) || 0.1);
  const threshold = falsePositiveCost / (falsePositiveCost + falseNegativeCost);
  $("optimal-threshold").textContent = threshold.toFixed(4);

  const stats = {};
  SERIES.forEach((series) => {
    stats[series.key] = decisionStats(
      currentAnalysis.yTrue,
      currentAnalysis.probabilities[series.key],
      falsePositiveCost,
      falseNegativeCost
    );
  });

  const body = document.querySelector("#decision-table tbody");
  body.textContent = "";
  SERIES.forEach((series) => {
    const item = stats[series.key];
    const delta = item.observedCost - stats.raw.observedCost;
    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = series.label;
    row.appendChild(heading);
    appendCells(row, [
      String(item.predictedPositive),
      String(item.falsePositive),
      String(item.falseNegative),
      item.expectedCost.toFixed(2),
      item.observedCost.toFixed(2),
      formatDelta(delta),
    ]);
    if (series.key !== "raw" && Math.abs(delta) >= 0.005) {
      row.lastElementChild.className = delta < 0 ? "delta-better" : "delta-worse";
    }
    body.appendChild(row);
  });
  syncUrl();
}

function currentBinning() {
  return document.querySelector('input[name="binning"]:checked').value;
}

function syncUrl() {
  const url = new URL(location.href);
  url.searchParams.set("model", $("model-select").value);
  url.searchParams.set("binning", currentBinning());
  url.searchParams.set("bins", $("bin-count").value);
  url.searchParams.set("fp", $("false-positive-cost").value);
  url.searchParams.set("fn", $("false-negative-cost").value);
  history.replaceState(null, "", url);
}

function restoreControlsFromUrl() {
  const params = new URLSearchParams(location.search);
  if (data.models[params.get("model")]) $("model-select").value = params.get("model");

  const bins = Number(params.get("bins"));
  if (bins >= 5 && bins <= 20) $("bin-count").value = String(Math.round(bins));

  const binning = params.get("binning");
  const binningControl = document.querySelector(`input[name="binning"][value="${binning}"]`);
  if (binningControl) binningControl.checked = true;

  for (const [parameter, id] of [["fp", "false-positive-cost"], ["fn", "false-negative-cost"]]) {
    const value = Number(params.get(parameter));
    if (value > 0) $(id).value = String(value);
  }
}

function render() {
  const binCount = Number($("bin-count").value);
  $("bin-count-value").textContent = String(binCount);
  const scheme = currentBinning();
  const metricSets = {};
  SERIES.forEach((series) => {
    metricSets[series.key] = metricsFor(
      currentAnalysis.yTrue,
      currentAnalysis.probabilities[series.key],
      binCount,
      scheme
    );
    console.assert(
      metricSets[series.key].reconstructionGap < 1e-9,
      `${series.label}: Murphy terms did not reconstruct Brier`
    );
  });
  renderChart(metricSets);
  renderReliabilityTable(metricSets);
  renderMetrics(metricSets);
  renderDecision();
}

function selectModel() {
  currentAnalysis = computeAnalysis($("model-select").value);
  $("temperature-value").textContent = `T = ${currentAnalysis.temperature.toFixed(4)}`;
  $("temperature-nll").textContent =
    `${currentAnalysis.calibrationNllRaw.toFixed(4)} to ${currentAnalysis.calibrationNllTemperature.toFixed(4)}`;
  $("isotonic-blocks").textContent = String(currentAnalysis.blocks.length);
  $("calibration-count").textContent = data.calibration.y_true.length.toLocaleString();
  $("fit-strip").hidden = false;
  $("status").textContent = `${currentAnalysis.model.label}, evaluated on ${data.test.y_true.length.toLocaleString()} untouched test cases.`;
  render();
}

function runSelfTests() {
  let passed = 0;
  let failed = 0;
  const check = (name, condition) => {
    console.assert(condition, `TrueOdds self-test failed: ${name}`);
    if (condition) passed += 1;
    else failed += 1;
  };

  const calibratedP = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75];
  const calibratedY = [0, 0, 0, 1, 0, 1, 1, 1];
  check("perfect calibration has ECE near zero", calibrationError(calibratedY, calibratedP, 4, "width").ece < 1e-12);

  check("constant 0.5 balanced predictor has Brier 0.25", Math.abs(brierScore([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.25) < 1e-12);

  const decomposition = murphyDecomposition(calibratedY, calibratedP);
  check("Murphy terms reconstruct Brier", Math.abs(brierScore(calibratedY, calibratedP) - decomposition.reconstructed) < 1e-9);

  // Hand-worked PAVA: labels 0,1,0,1 pool the middle violation to 0.5,0.5.
  const pavaBlocks = fitIsotonic([1, 2, 3, 4], [0, 1, 0, 1]);
  const pavaOutput = applyIsotonic([1, 2, 3, 4], pavaBlocks);
  check("PAVA is monotone and matches hand example",
    pavaOutput.every((value, index) => index === 0 || value >= pavaOutput[index - 1]) &&
    pavaOutput.every((value, index) => Math.abs(value - [0, 0.5, 0.5, 1][index]) < 1e-12));

  const identityInput = [0.1, 0.5, 0.9];
  const identityOutput = temperatureScale(identityInput, 1);
  const overconfidentP = [0.02, 0.15, 0.9, 0.1, 0.85, 0.98];
  const overconfidentY = [0, 0, 0, 1, 1, 1];
  const fittedTemperature = fitTemperature(overconfidentP, overconfidentY);
  check("temperature identity and fitted NLL reduction",
    identityOutput.every((value, index) => Math.abs(value - identityInput[index]) < 1e-12) &&
    binaryNll(overconfidentY, temperatureScale(overconfidentP, fittedTemperature)) <
      binaryNll(overconfidentY, overconfidentP));

  const tieFreeP = [0.1, 0.2, 0.8, 0.9];
  const tieFreeY = [0, 0, 1, 1];
  const isotonicP = applyIsotonic(tieFreeP, fitIsotonic(tieFreeP, tieFreeY));
  check("AUC invariant after isotonic on tie-free ordered case",
    Math.abs(aucByRank(tieFreeY, tieFreeP) - aucByRank(tieFreeY, isotonicP)) < 1e-12);

  const summary = `TrueOdds self-test: ${passed}/${passed + failed} passed`;
  if (failed) console.error(summary);
  else console.log(summary);
  return failed === 0;
}

async function init() {
  try {
    const response = await fetch("data/adult.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
    restoreControlsFromUrl();
    selectModel();
  } catch (error) {
    $("status").textContent = `Could not load data/adult.json: ${error.message}. Serve the project over HTTP and rebuild the data artifact if needed.`;
    $("status").classList.add("error");
    return;
  }

  $("model-select").addEventListener("change", selectModel);
  $("bin-count").addEventListener("input", render);
  document.querySelectorAll('input[name="binning"]').forEach((input) => input.addEventListener("change", render));
  $("false-positive-cost").addEventListener("input", renderDecision);
  $("false-negative-cost").addEventListener("input", renderDecision);

  if (new URLSearchParams(location.search).get("selftest") === "1") runSelfTests();
}

if (hasDOM) document.addEventListener("DOMContentLoaded", init);
if (!hasDOM && typeof module !== "undefined" && require.main === module) {
  process.exitCode = runSelfTests() ? 0 : 1;
}
if (typeof module !== "undefined") {
  module.exports = {
    applyIsotonic,
    aucByRank,
    binaryNll,
    calibrationError,
    decisionStats,
    fitIsotonic,
    fitTemperature,
    metricsFor,
    murphyDecomposition,
    temperatureScale,
  };
}
