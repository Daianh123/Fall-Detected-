const form = document.querySelector("#source-form");
const tabs = [...document.querySelectorAll("[data-source-tab]")];
const panels = [...document.querySelectorAll("[data-source-panel]")];
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const message = document.querySelector("#form-message");
const videoFeed = document.querySelector("#video-feed");
const feedEmpty = document.querySelector("#feed-empty");
const feedLive = document.querySelector("#feed-live");
const systemDot = document.querySelector("#system-dot");
const systemState = document.querySelector("#system-state");
const resultMark = document.querySelector("#result-mark");

let sourceType = "rtsp";
let statusTimer = null;

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    sourceType = tab.dataset.sourceTab;
    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.sourcePanel !== sourceType;
    });
    message.textContent = "";
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "Đang khởi tạo mô hình…";
  startButton.disabled = true;

  try {
    let body;
    if (sourceType === "file") {
      const file = document.querySelector("#video-file").files[0];
      if (!file) throw new Error("Hãy chọn một tệp video trước.");
      body = new FormData();
      body.append("video", file);
    } else {
      body = new FormData();
      body.append(
        "source",
        document.querySelector("#rtsp-source").value.trim(),
      );
    }

    const response = await fetch("/api/start", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok)
      throw new Error(payload.error || "Không thể bắt đầu phát hiện.");
    message.textContent = "Đã nhận nguồn. Đang kết nối…";
    stopButton.disabled = false;
    feedEmpty.hidden = true;
    feedLive.hidden = false;
    videoFeed.src = `/video_feed?t=${Date.now()}`;
    beginStatusUpdates();
  } catch (error) {
    message.textContent = error.message;
    startButton.disabled = false;
  }
});

stopButton.addEventListener("click", async () => {
  await fetch("/api/stop", { method: "POST" });
  message.textContent = "Đã gửi yêu cầu dừng.";
  stopButton.disabled = true;
});

function beginStatusUpdates() {
  window.clearInterval(statusTimer);
  statusTimer = window.setInterval(refreshStatus, 1000);
  refreshStatus();
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const state = await response.json();
    systemState.textContent = state.error || state.label;
    systemDot.classList.toggle("is-active", state.running);
    systemDot.classList.toggle(
      "is-alert",
      state.phase === "alert" || state.phase === "error",
    );
    document.querySelector("#feed-source").textContent =
      state.source || "CHƯA CÓ NGUỒN VIDEO";
    document.querySelector("#frame-count").textContent =
      `${state.frames} KHUNG HÌNH`;
    document.querySelector("#result-label").textContent = state.label;
    document.querySelector("#result-confidence").textContent =
      state.confidence === null
        ? state.error || "Kết quả sẽ xuất hiện tại đây"
        : `Độ tin cậy ${(state.confidence * 100).toFixed(1)}%`;
    resultMark.className = `result-mark ${state.phase === "alert" ? "alert" : state.phase === "monitoring" ? "normal" : "neutral"}`;

    if (!state.running) {
      window.clearInterval(statusTimer);
      startButton.disabled = false;
      stopButton.disabled = true;
      feedLive.hidden = true;
      if (
        state.phase === "error" ||
        state.phase === "complete" ||
        state.phase === "stopped"
      ) {
        message.textContent =
          state.error ||
          (state.phase === "complete" ? "Đã xử lý hết video." : "");
      }
      if (!videoFeed.getAttribute("src") || state.phase === "error") {
        feedEmpty.hidden = false;
        videoFeed.removeAttribute("src");
      }
    }
  } catch {
    systemState.textContent = "Mất kết nối tới máy chủ";
  }
}
