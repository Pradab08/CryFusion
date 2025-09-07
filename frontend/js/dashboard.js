document.addEventListener("DOMContentLoaded", () => {
  let cryChart;
  let ws = null;

  // Initialize cry frequency chart
  function initCryChart() {
    const ctx = document.getElementById("cryChart");
    if (!ctx) return;

    cryChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Cry Frequency (per hour)',
          data: [],
          backgroundColor: 'rgba(54,162,235,0.2)',
          borderColor: 'rgba(54,162,235,1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: true, max: 10 }
        },
        animation: { duration: 750 }
      }
    });
  }

  // Update dashboard stats
  function updateStats(data) {
    const totalCries = document.getElementById("total-cries");
    const heartRate = document.getElementById("heart-rate");
    const temperature = document.getElementById("temperature");
    const diaperStatus = document.getElementById("diaper-status");
    const lastCryType = document.getElementById("last-cry-type");
    const lastCryTime = document.getElementById("last-cry-time");

    if (totalCries) totalCries.textContent = data.total_cries || 0;
    if (heartRate) heartRate.textContent = (data.heart_rate || 0) + " bpm";
    if (temperature) temperature.textContent = (data.temperature || 0) + " °C";
    if (diaperStatus) diaperStatus.textContent = data.diaper_status || "Unknown";
    if (lastCryType) lastCryType.textContent = data.last_cry_type || "None";
    if (lastCryTime) lastCryTime.textContent = data.last_cry_time || "N/A";
  }

  // Update cry chart with new data
  function updateCryChart(hour, count) {
    if (!cryChart) return;

    const timeStr = hour.toString().padStart(2, '0') + ':00';
    
    // Add new data point
    cryChart.data.labels.push(timeStr);
    cryChart.data.datasets[0].data.push(count);

    // Keep only last 24 hours
    if (cryChart.data.labels.length > 24) {
      cryChart.data.labels.shift();
      cryChart.data.datasets[0].data.shift();
    }

    cryChart.update('none');
  }

  // Connect to dashboard WebSocket for real-time updates
  function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket('ws://localhost:8000/ws/dashboard');
    
    ws.onopen = () => {
      console.log('Dashboard WebSocket connected');
      // Load initial data
      loadDashboardData();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'dashboard_update') {
          updateStats(data);
          if (data.cry_data) {
            updateCryChart(data.cry_data.hour, data.cry_data.count);
          }
        }
      } catch (e) {
        console.warn('Invalid WebSocket message:', event.data);
      }
    };

    ws.onclose = () => {
      console.log('Dashboard WebSocket disconnected, reconnecting...');
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
      console.error('Dashboard WebSocket error:', error);
    };
  }

  // Load initial dashboard data
  async function loadDashboardData() {
    try {
      // Fetch stats
      const resStats = await fetch('http://localhost:8000/api/dashboard/stats');
      const stats = await resStats.json();
      updateStats(stats);

      // Fetch trends
      const resTrends = await fetch('http://localhost:8000/api/dashboard/trends');
      const trends = await resTrends.json();
      
      if (trends.labels && trends.data) {
        trends.labels.forEach((label, i) => {
          updateCryChart(parseInt(label.split(':')[0]), trends.data[i]);
        });
      }
    } catch(err) {
      console.error('Failed to load dashboard data:', err);
    }
  }

  // Initialize
  initCryChart();
  connectWebSocket();

  // Fallback polling if WebSocket fails
  setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      loadDashboardData();
    }
  }, 15000);
});
