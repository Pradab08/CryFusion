document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("audioFile");
  const resultBox = document.getElementById("resultBox");
  const confidenceFill = document.getElementById("confidenceFill");
  const probsList = document.getElementById("probsList");
  const gradCamImg = document.getElementById("gradCamImg");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) {
      alert("Please choose an audio file");
      return;
    }
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    resultBox.textContent = "Analyzing...";
    try {
      const res = await fetch('http://localhost:8000/api/cry_analysis/predict', { method: 'POST', body: formData });
      const data = await res.json();
      const pct = Math.round(data.confidence * 100);
      resultBox.innerHTML = `Prediction: <span class="badge bg-primary">${data.prediction}</span>`;
      confidenceFill.style.width = `${pct}%`;
      confidenceFill.textContent = `${pct}%`;
      confidenceFill.className = `progress-bar ${pct >= 75 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger'}`;
      probsList.innerHTML = Object.entries(data.probs || {})
        .sort((a,b)=>b[1]-a[1])
        .map(([label, p])=>`<li class="list-group-item d-flex justify-content-between align-items-center">${label}<span class="badge bg-secondary">${Math.round(p*100)}%</span></li>`)
        .join("");
      if (data.grad_cam) {
      gradCamImg.src = `data:image/png;base64,${data.grad_cam}`;
      }
    } catch (err) {
      console.error(err);
      resultBox.textContent = "Error during analysis.";
    }
  });

  // Live mic streaming over WebSocket
  const startBtn = document.getElementById("startRecord");
  const stopBtn = document.getElementById("stopRecord");
  const audioEl = document.getElementById("recordedAudio");
  let mediaRecorder = null;
  let ws = null;

  // Generate live heatmap for real-time display
  function generateLiveHeatmap(label, confidence) {
    // Create a canvas-based heatmap
    const canvas = document.createElement('canvas');
    canvas.width = 200;
    canvas.height = 150;
    const ctx = canvas.getContext('2d');
    
    // Generate heatmap based on label and confidence
    const colors = {
      'hungry': '#ff6b6b',
      'tired': '#4ecdc4', 
      'discomfort': '#45b7d1',
      'belly_pain': '#96ceb4',
      'burping': '#feca57'
    };
    
    const baseColor = colors[label] || '#6c5ce7';
    const intensity = Math.min(confidence, 0.95);
    
    // Create gradient heatmap
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, baseColor + Math.floor(intensity * 255).toString(16).padStart(2, '0'));
    gradient.addColorStop(0.5, baseColor + '80');
    gradient.addColorStop(1, baseColor + '40');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add label text
    ctx.fillStyle = 'white';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(label.toUpperCase(), canvas.width/2, canvas.height/2);
    ctx.font = '12px Arial';
    ctx.fillText(`${Math.round(confidence * 100)}%`, canvas.width/2, canvas.height/2 + 20);
    
    // Convert to data URL and display
    gradCamImg.src = canvas.toDataURL();
    gradCamImg.style.display = 'block';
  }

  // Update WebSocket connection to show live predictions
  function openWs() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return ws;
    ws = new WebSocket("ws://localhost:8000/ws/audio");
    ws.onopen = () => {
      resultBox.textContent = "Microphone streaming started...";
      // Show live recording indicator
      resultBox.innerHTML = '<div class="d-flex align-items-center"><i class="bi bi-mic-fill text-danger me-2"></i>Live recording - analyzing audio...</div>';
    };
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'prediction') {
          const pct = Math.round(msg.confidence * 100);
          resultBox.innerHTML = `Live: <span class="badge bg-primary">${msg.label}</span>`;
          confidenceFill.style.width = `${pct}%`;
          confidenceFill.textContent = `${pct}%`;
          confidenceFill.className = `progress-bar ${pct >= 75 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger'}`;
          if (msg.probs) {
            probsList.innerHTML = Object.entries(msg.probs)
              .sort((a,b)=>b[1]-a[1])
              .map(([label, p])=>`<li class="list-group-item d-flex justify-content-between align-items-center">${label}<span class="badge bg-secondary">${Math.round(p*100)}%</span></li>`)
              .join("");
          }
          
          // Show live Grad-CAM if available
          if (msg.grad_cam) {
            gradCamImg.src = `data:image/png;base64,${msg.grad_cam}`;
            gradCamImg.style.display = 'block';
          } else {
            // Generate placeholder heatmap for live display
            generateLiveHeatmap(msg.label, msg.confidence);
          }
        }
      } catch (e) {
        console.warn("ws message", evt.data);
      }
    };
    ws.onclose = () => {
      if (resultBox.textContent.includes("Microphone")) {
        resultBox.textContent = "Microphone streaming stopped.";
      }
    };
    ws.onerror = () => {
      resultBox.textContent = "WebSocket error";
    };
    return ws;
  }

  startBtn.addEventListener("click", async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    const chunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data);
      }
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      audioEl.src = URL.createObjectURL(blob);
    };
    mediaRecorder.start(500); // collect small chunks

    const socket = openWs();

    // periodic sender: convert recent chunks to WAV and send
    const sendInterval = setInterval(async () => {
      if (!chunks.length || !socket || socket.readyState !== WebSocket.OPEN) return;
      const batch = new Blob(chunks.splice(0, chunks.length), { type: 'audio/webm' });
      const wav = await webmToWav(batch);
      socket.send(await wav.arrayBuffer());
    }, 600);

    startBtn.disabled = true;
    stopBtn.disabled = false;

    stopBtn.onclick = () => {
      clearInterval(sendInterval);
      try { mediaRecorder.stop(); } catch {}
      startBtn.disabled = false;
      stopBtn.disabled = true;
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    };
  });

  async function webmToWav(blob) {
    // Use OfflineAudioContext to decode then encode PCM as WAV
    const arrayBuf = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuf.slice(0));
    const numChannels = 1;
    const length = decoded.length;
    const sampleRate = 22050;
    // resample to 22050 by drawing into OfflineAudioContext
    const offline = new OfflineAudioContext(numChannels, Math.ceil(length * (sampleRate/decoded.sampleRate)), sampleRate);
    const src = offline.createBufferSource();
    const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    // mixdown to mono
    const ch0 = mono.getChannelData(0);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const data = decoded.getChannelData(c);
      for (let i = 0; i < data.length; i++) ch0[i] = (ch0[i] + data[i]) / (c + 1);
    }
    src.buffer = mono;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    const pcm = rendered.getChannelData(0);
    return encodeWav(pcm, rendered.sampleRate);
  }

  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    const numChannels = 1;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * bytesPerSample, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 8 * bytesPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * bytesPerSample, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }
});
