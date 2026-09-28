"use strict";
document.querySelector("#error-filter").addEventListener("change", (event) => {
  document.querySelectorAll(".error").forEach((item) => {
    item.hidden = event.target.value !== "all" && item.dataset.label !== event.target.value;
  });
});
document.querySelector("#predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = document.querySelector("#prediction");
  const button = event.target.querySelector("button");
  if (location.protocol === "file:") {
    status.textContent = "请在项目目录运行 sentiment-lab serve，然后打开 http://127.0.0.1:8765。";
    return;
  }
  button.disabled = true;
  status.textContent = "正在分析…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: document.querySelector("#review").value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "预测请求失败");
    status.textContent = `预测：${data.label === 1 ? "正面" : "负面"} · 模型：${data.model} · ${data.score_type}：${data.score.toFixed(4)}（未校准）`;
  } catch (error) {
    status.textContent = `预测未完成：${error.message}。请确认本地 sentiment-lab serve 正在运行。`;
  } finally { button.disabled = false; }
});
