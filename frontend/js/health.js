document.addEventListener("DOMContentLoaded", () => {
  const heartRate = document.getElementById("heartRate");
  const temperature = document.getElementById("temperature");
  const diaperStatus = document.getElementById("diaperStatus");
  const alertsContainer = document.getElementById("alertsContainer");

  let hrChart, tempChart;
  let ws = null;

  // Initialize charts
  function initCharts() {
    const hrCtx = document.getElementById("heartRateChart").getContext("2d");
    const tempCtx = document.getElementById("temperatureChart").getContext("2d");

    hrChart = new Chart(hrCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{ 
          label: 'Heart Rate', 
          data: [], 
          borderColor: 'rgba(255,0,0,1)', 
          backgroundColor: 'rgba(255,0,0,0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: false, min: 80, max: 140 }
        },
        animation: { duration: 750 }
      }
    });

    tempChart = new Chart(tempCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{ 
          label: 'Temperature', 
          data: [], 
          borderColor: 'rgba(0,128,0,1)', 
          backgroundColor: 'rgba(0,128,0,0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: false, min: 36, max: 39 }
        },
        animation: { duration: 750 }
      }
    });
  }

  function updateSensorCards(hr, temp, diaper) {
    heartRate.textContent = hr + " bpm";
    temperature.textContent = temp + " °C";
    diaperStatus.textContent = diaper;

    // Update card colors based on values
    const hrCard = heartRate.closest('.card');
    const tempCard = temperature.closest('.card');
    const diaperCard = diaperStatus.closest('.card');

    // Heart rate status
    if (hr > 120) {
      hrCard.className = 'card text-white bg-danger';
      heartRate.innerHTML = hr + ' bpm <i class="bi bi-exclamation-triangle"></i>';
    } else if (hr > 100) {
      hrCard.className = 'card text-white bg-warning';
      heartRate.innerHTML = hr + ' bpm <i class="bi bi-exclamation-circle"></i>';
    } else {
      hrCard.className = 'card text-white bg-primary';
      heartRate.innerHTML = hr + ' bpm <i class="bi bi-heart-pulse"></i>';
    }

    // Temperature status
    if (temp > 38) {
      tempCard.className = 'card text-white bg-danger';
      temperature.innerHTML = temp + ' °C <i class="bi bi-thermometer-high"></i>';
    } else if (temp > 37.5) {
      tempCard.className = 'card text-white bg-warning';
      temperature.innerHTML = temp + ' °C <i class="bi bi-thermometer-half"></i>';
    } else {
      tempCard.className = 'card text-white bg-success';
      temperature.innerHTML = temp + ' °C <i class="bi bi-thermometer-low"></i>';
    }

    // Diaper status
    if (diaper === "Wet") {
      diaperCard.className = 'card text-white bg-warning';
      diaperStatus.innerHTML = 'Wet <i class="bi bi-droplet-fill"></i>';
    } else {
      diaperCard.className = 'card text-white bg-success';
      diaperStatus.innerHTML = 'Dry <i class="bi bi-check-circle"></i>';
    }

    // Update alerts
    const alerts = [];
    if (hr > 120) alerts.push("⚠️ Heart rate high!");
    if (temp > 38) alerts.push("🔥 Temperature high!");
    if (diaper === "Wet") alerts.push("💧 Diaper needs changing");
    
    alertsContainer.innerHTML = alerts.length ? 
      alerts.map(a => `<p class="text-danger fw-bold">${a}</p>`).join('') : 
      "<p class='text-success'>✅ All vitals normal</p>";
  }

  function updateCharts(hr, temp, timestamp) {
    const timeStr = timestamp.toLocaleTimeString('en-US', { hour12: false });
    
    // Add new data points
    hrChart.data.labels.push(timeStr);
    hrChart.data.datasets[0].data.push(hr);
    
    tempChart.data.labels.push(timeStr);
    tempChart.data.datasets[0].data.push(temp);

    // Keep only last 20 data points
    if (hrChart.data.labels.length > 20) {
      hrChart.data.labels.shift();
      hrChart.data.datasets[0].data.shift();
    }
    if (tempChart.data.labels.length > 20) {
      tempChart.data.labels.shift();
      tempChart.data.datasets[0].data.shift();
    }

    // Update charts
    hrChart.update('none');
    tempChart.update('none');
  }

  function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket('ws://localhost:8000/ws/sensor');
    
    ws.onopen = () => {
      console.log('WebSocket connected to sensor stream');
      // Request initial data
      fetch('http://localhost:8000/api/health/sensors')
        .then(res => res.json())
        .then(data => {
          updateSensorCards(data.heart_rate, data.temperature, data.diaper_status);
          // Initialize charts with historical data
          if (data.trends) {
            data.trends.time.forEach((time, i) => {
              const hr = data.trends.heart_rate[i];
              const temp = data.trends.temperature[i];
              const date = new Date();
              date.setHours(parseInt(time.split(':')[0]), parseInt(time.split(':')[1]));
              updateCharts(hr, temp, date);
            });
          }
        });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'sensor_update') {
          updateSensorCards(data.heart_rate, data.temperature, data.diaper_status);
          updateCharts(data.heart_rate, data.temperature, new Date());
        }
      } catch (e) {
        console.warn('Invalid WebSocket message:', event.data);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  // Initialize
  initCharts();
  connectWebSocket();

  // Fallback: poll every 10 seconds if WebSocket fails
  setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      fetch('http://localhost:8000/api/health/sensors')
        .then(res => res.json())
        .then(data => {
          updateSensorCards(data.heart_rate, data.temperature, data.diaper_status);
        })
        .catch(err => console.error('Fallback fetch failed:', err));
    }
  }, 10000);
});
