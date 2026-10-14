const LOCAL_SPLIT = location.port === '5500';
const API_BASE = LOCAL_SPLIT ? 'http://localhost:8000/api/predict' : '/api/predict';

// ---------- Nav ----------
function showView(name) {
  document.querySelectorAll('.nav-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.view === name));
  document.querySelectorAll('.view').forEach(v =>
    v.classList.toggle('active', v.id === name));
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => showView(btn.dataset.view));
});

document.getElementById('brandLink').addEventListener('click', e => {
  e.preventDefault();
  showView('overview');
});

// ---------- Input mode toggle ----------
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.mode + 'Mode').classList.add('active');
  });
});

function tierClass(label) {
  return 'tier-' + label.toLowerCase().replace(/\s+/g, '-');
}

function formatMoney(value) {
  return '£' + Math.round(value).toLocaleString();
}

// ---------- Dashboard data ----------
async function loadDashboard() {
  try {
    const res = await fetch('data/dashboard.json');
    const data = await res.json();
    renderHero(data);
    renderLedger(data);
    renderCharts(data);
  } catch (err) {
    document.getElementById('heroStats').innerHTML =
      '<div class="metric-card"><span class="metric-label">Could not load report data.</span></div>';
  }
}

function renderHero(data) {
  document.getElementById('heroStats').innerHTML = `
    <div class="metric-card">
      <strong class="metric-value">${data.total_customers.toLocaleString()}</strong>
      <span class="metric-label">accounts analyzed</span>
    </div>
    <div class="metric-card">
      <strong class="metric-value">${formatMoney(data.total_monetary)}</strong>
      <span class="metric-label">in lifetime value tracked</span>
    </div>
  `;

  const tbody = document.querySelector('#heroLedger tbody');
  const sorted = [...data.cluster_summary].sort((a, b) => b.monetary - a.monetary);
  tbody.innerHTML = sorted.map(c => `
    <tr class="${tierClass(c.label)}">
      <td>${c.label}</td>
      <td>${c.count.toLocaleString()} accts</td>
      <td>${c.recency}d &middot; ${c.frequency}x &middot; ${formatMoney(c.monetary)}</td>
    </tr>
  `).join('');
}

function renderLedger(data) {
  const tbody = document.querySelector('#clusterLedger tbody');
  const sorted = [...data.cluster_summary].sort((a, b) => b.monetary - a.monetary);
  tbody.innerHTML = sorted.map(c => `
    <tr class="${tierClass(c.label)}">
      <td>${c.label}</td>
      <td>${c.count.toLocaleString()}</td>
      <td>${c.recency}d</td>
      <td>${c.frequency}x</td>
      <td>${formatMoney(c.monetary)}</td>
    </tr>
  `).join('');
}

function barChart(canvasId, counts, colors) {
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: Object.keys(counts),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
        barThickness: 24,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 900, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#252521',
          bodyFont: { family: 'IBM Plex Mono' },
          padding: 12,
          displayColors: false,
        },
      },
      scales: {
        x: { grid: { display: false }, border: { display: false }, ticks: { font: { size: 10 }, maxRotation: 0 } },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(128,124,112,0.15)' },
          border: { display: false },
          ticks: { font: { family: 'IBM Plex Mono', size: 9 } },
        },
      },
    },
  });
}

function renderCharts(data) {
  Chart.defaults.font.family = 'IBM Plex Sans';
  Chart.defaults.color = '#858078';
  barChart('clusterChart', data.cluster_counts, ['#43583E', '#708668', '#B28C50', '#7B3030']);
  barChart('segmentChart', data.segment_counts, '#B28C50');
}

loadDashboard();

// ---------- Predictor ----------
function renderRecommendationBlock(label, rec) {
  if (!rec) return '';
  return `
    <p class="stamp-profile">${rec.profile}</p>
    <ul class="stamp-actions">
      ${rec.actions.map(a => `<li>${a}</li>`).join('')}
    </ul>
    <p class="stamp-priority">${rec.priority}</p>
  `;
}

function renderSingleResult(label, rec) {
  document.getElementById('predictionResult').innerHTML = `
    <div class="stamp-card">
      <div class="stamp ${tierClass(label)}">${label}</div>
      ${renderRecommendationBlock(label, rec)}
    </div>
  `;
}

function renderBulkResult(customers) {
  const rows = customers.map(c => `
    <tr class="${tierClass(c.cluster_label)}">
      <td>${c.customer_id}</td>
      <td>${c.recency}d</td>
      <td>${c.frequency}x</td>
      <td>${formatMoney(c.monetary)}</td>
      <td>${c.cluster_label}</td>
    </tr>
  `).join('');

  const uniqueLabels = [...new Set(customers.map(c => c.cluster_label))];
  const recBlocks = uniqueLabels.map(label => {
    const rec = customers.find(c => c.cluster_label === label).recommendation;
    return `
      <div class="stamp-card">
        <div class="stamp ${tierClass(label)}">${label}</div>
        ${renderRecommendationBlock(label, rec)}
      </div>
    `;
  }).join('');

  document.getElementById('predictionResult').innerHTML = `
    <div class="card">
      <div class="table-scroll">
        <table class="ledger bulk-table">
          <thead><tr><th>Customer</th><th>Recency</th><th>Frequency</th><th>Monetary</th><th>Segment</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
    <div class="rec-grid">${recBlocks}</div>
  `;
}

function renderError(message) {
  document.getElementById('predictionResult').innerHTML =
    `<div class="error-card">${message}</div>`;
}

document.getElementById('predictRfmBtn').addEventListener('click', async () => {
  const payload = {
    recency: parseFloat(document.getElementById('recencyInput').value),
    frequency: parseFloat(document.getElementById('frequencyInput').value),
    monetary: parseFloat(document.getElementById('monetaryInput').value),
  };

  try {
    const res = await fetch(`${API_BASE}/rfm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Lookup failed.');
    const data = await res.json();
    renderSingleResult(data.cluster_label, data.recommendation);
  } catch (err) {
    renderError(err.message);
  }
});

document.getElementById('predictUploadBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('csvFileInput');
  if (!fileInput.files.length) {
    renderError('Choose a CSV file first.');
    return;
  }

  const text = await fileInput.files[0].text();
  const rows = text.trim().split('\n');
  const headers = rows[0].split(',').map(h => h.trim());
  const transactions = rows.slice(1).map(row => {
    const values = row.split(',');
    const obj = {};
    headers.forEach((h, i) => { obj[h] = values[i] ? values[i].trim() : ''; });
    obj.Quantity = parseFloat(obj.Quantity);
    obj.UnitPrice = parseFloat(obj.UnitPrice);
    return obj;
  });

  try {
    const res = await fetch(`${API_BASE}/transactions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transactions }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Lookup failed.');
    const data = await res.json();
    renderBulkResult(data.customers);
  } catch (err) {
    renderError(err.message);
  }
});