import './style.css';

const stages = [
  {
    formula: '[1, 28, 28] → [49, 16]',
    title: 'Potong gambar',
    description: 'Gambar abu-abu 28 × 28 dibagi menjadi kisi 7 × 7. Tiap patch memuat 16 nilai piksel, lalu disusun berurutan dari kiri atas.',
    code: 'Lihat patchify() di model.py.',
  },
  {
    formula: '[49, 16] → [50, 64]',
    title: 'Bentuk token',
    description: 'Proyeksi linear mengubah setiap patch menjadi vektor 64 angka. Model menambahkan satu token kelas dan posisi yang dipelajari agar urutan patch tetap diketahui.',
    code: 'Lihat PatchEmbedding dan TinyViT.forward() di model.py.',
  },
  {
    formula: '[50, 64] → [4, 50, 50]',
    title: 'Hubungkan token',
    description: 'Setiap head membentuk query, key, dan value. Skor query terhadap semua key dinormalisasi menjadi bobot attention, lalu dipakai untuk menggabungkan value.',
    code: 'Lihat MultiHeadSelfAttention di model.py.',
  },
  {
    formula: '[50, 64] → [10]',
    title: 'Baca kelas',
    description: 'Setelah dua encoder block, vektor token kelas masuk ke lapis linear yang menghasilkan sepuluh logit. Softmax mengubah logit menjadi probabilitas.',
    code: 'Lihat TinyViT.classifier di model.py.',
  },
];

const elements = {
  heroCanvas: document.querySelector('#hero-canvas'),
  sampleCanvas: document.querySelector('#sample-canvas'),
  attentionCanvas: document.querySelector('#attention-canvas'),
  sampleList: document.querySelector('#sample-list'),
  querySelect: document.querySelector('#query-select'),
  layerSelect: document.querySelector('#layer-select'),
  headSelect: document.querySelector('#head-select'),
  sampleIndex: document.querySelector('#sample-index'),
  imageCaption: document.querySelector('#image-caption'),
  predictionLabel: document.querySelector('#prediction-label'),
  probabilityList: document.querySelector('#probability-list'),
  attentionSummary: document.querySelector('#attention-summary'),
  resultsContent: document.querySelector('#results-content'),
  lossChart: document.querySelector('#loss-chart'),
  loadState: document.querySelector('#load-state'),
  emptyState: document.querySelector('#empty-state'),
  errorState: document.querySelector('#error-state'),
  labContent: document.querySelector('#lab-content'),
};

const state = { data: null, sample: 0, query: 0, layer: 1, head: 0 };

function drawImage(canvas, pixels, grid = false, selectedPatch = -1) {
  const context = canvas.getContext('2d');
  const scale = canvas.width / 28;
  context.clearRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < 28; row += 1) {
    for (let column = 0; column < 28; column += 1) {
      const value = 255 - pixels[row * 28 + column];
      context.fillStyle = `rgb(${value}, ${value}, ${value})`;
      context.fillRect(column * scale, row * scale, scale + 0.2, scale + 0.2);
    }
  }
  if (grid) {
    context.strokeStyle = 'rgba(32, 43, 42, 0.5)';
    context.lineWidth = 1;
    for (let index = 1; index < 7; index += 1) {
      const point = index * 4 * scale;
      context.beginPath();
      context.moveTo(point, 0);
      context.lineTo(point, canvas.height);
      context.moveTo(0, point);
      context.lineTo(canvas.width, point);
      context.stroke();
    }
  }
  if (selectedPatch >= 0) {
    const row = Math.floor(selectedPatch / 7);
    const column = selectedPatch % 7;
    context.strokeStyle = '#a23e20';
    context.lineWidth = 4;
    context.strokeRect(column * 4 * scale + 2, row * 4 * scale + 2, 4 * scale - 4, 4 * scale - 4);
  }
}

function drawPlaceholder() {
  const canvas = elements.heroCanvas;
  const context = canvas.getContext('2d');
  context.fillStyle = '#f6f3ec';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = '#a23e20';
  context.lineWidth = 1;
  for (let index = 0; index <= 7; index += 1) {
    const point = index * canvas.width / 7;
    context.beginPath();
    context.moveTo(point, 0);
    context.lineTo(point, canvas.height);
    context.moveTo(0, point);
    context.lineTo(canvas.width, point);
    context.stroke();
  }
}

function drawAttention(weights) {
  const canvas = elements.attentionCanvas;
  const context = canvas.getContext('2d');
  const patchWeights = weights.slice(1);
  const maximum = Math.max(...patchWeights, 0.000001);
  const size = canvas.width / 7;
  patchWeights.forEach((weight, index) => {
    const strength = Math.sqrt(weight / maximum);
    const start = [250, 240, 233];
    const end = [162, 62, 32];
    const color = start.map((part, channel) => Math.round(part + (end[channel] - part) * strength));
    context.fillStyle = `rgb(${color.join(',')})`;
    context.fillRect((index % 7) * size, Math.floor(index / 7) * size, size, size);
    context.strokeStyle = 'rgba(32, 43, 42, 0.35)';
    context.lineWidth = 1;
    context.strokeRect((index % 7) * size + 0.5, Math.floor(index / 7) * size + 0.5, size, size);
  });
  if (state.query > 0) {
    const index = state.query - 1;
    context.strokeStyle = '#202b2a';
    context.lineWidth = 4;
    context.strokeRect((index % 7) * size + 2, Math.floor(index / 7) * size + 2, size - 4, size - 4);
  }
}

function addOptions(select, labels) {
  select.replaceChildren();
  labels.forEach((label, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = label;
    select.append(option);
  });
}

function renderSamples() {
  elements.sampleList.replaceChildren();
  state.data.samples.forEach((sample, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `sample-button${index === state.sample ? ' is-active' : ''}`;
    button.setAttribute('aria-pressed', String(index === state.sample));
    button.setAttribute('aria-label', `Contoh ${state.data.classes[sample.label]}, indeks uji ${sample.test_index}`);
    const thumbnail = document.createElement('canvas');
    thumbnail.width = 28;
    thumbnail.height = 28;
    thumbnail.setAttribute('aria-hidden', 'true');
    drawImage(thumbnail, sample.pixels);
    const label = document.createElement('span');
    label.textContent = state.data.classes[sample.label];
    button.append(thumbnail, label);
    button.addEventListener('click', () => {
      state.sample = index;
      renderSamples();
      renderSample();
    });
    elements.sampleList.append(button);
  });
}

function renderPrediction(sample) {
  const name = state.data.classes[sample.prediction];
  const correct = sample.prediction === sample.label;
  elements.predictionLabel.replaceChildren();
  elements.predictionLabel.append(document.createTextNode(name));
  const status = document.createElement('span');
  status.textContent = correct ? 'Prediksi sesuai label uji' : `Prediksi berbeda dari label: ${state.data.classes[sample.label]}`;
  elements.predictionLabel.append(status);
  elements.probabilityList.replaceChildren();
  const ranking = sample.probabilities.map((value, index) => ({ index, value }))
    .sort((first, second) => second.value - first.value).slice(0, 3);
  ranking.forEach(({ index, value }) => {
    const row = document.createElement('div');
    row.className = 'probability-row';
    const label = document.createElement('span');
    label.textContent = state.data.classes[index];
    const number = document.createElement('strong');
    number.textContent = `${(value * 100).toFixed(1)}%`;
    const track = document.createElement('div');
    track.className = 'probability-track';
    const fill = document.createElement('span');
    fill.className = 'probability-fill';
    fill.style.width = `${value * 100}%`;
    track.append(fill);
    row.append(label, number, track);
    elements.probabilityList.append(row);
  });
}

function renderAttention() {
  const sample = state.data.samples[state.sample];
  const weights = sample.attention[state.layer][state.head][state.query];
  drawAttention(weights);
  const patches = weights.slice(1);
  const maximum = Math.max(...patches);
  const strongest = patches.indexOf(maximum) + 1;
  const queryName = state.query === 0 ? 'Token kelas' : `Patch ${state.query}`;
  elements.attentionSummary.textContent = `${queryName} memberi bobot terbesar pada patch ${strongest}: ${maximum.toFixed(3)}. Bobot ke token kelas: ${weights[0].toFixed(3)}.`;
  elements.querySelect.value = String(state.query);
  elements.layerSelect.value = String(state.layer);
  elements.headSelect.value = String(state.head);
}

function renderSample() {
  const sample = state.data.samples[state.sample];
  drawImage(elements.heroCanvas, sample.pixels, true);
  drawImage(elements.sampleCanvas, sample.pixels, true, state.query - 1);
  elements.sampleIndex.textContent = `indeks uji ${sample.test_index}`;
  elements.imageCaption.replaceChildren();
  elements.imageCaption.append(document.createTextNode(`Label: ${state.data.classes[sample.label]}`));
  const subline = document.createElement('span');
  subline.textContent = 'Gambar asli 28 × 28 piksel, diperbesar agar patch terlihat.';
  elements.imageCaption.append(subline);
  renderPrediction(sample);
  renderAttention();
}

function renderResults() {
  const { vit, cnn } = state.data.metrics;
  const formatNumber = (value) => new Intl.NumberFormat('id-ID').format(value);
  const formatMinutes = (value) => `${(value / 60).toFixed(1)} menit`;
  const accuracy = (value) => `${(value * 100).toFixed(1)}%`;
  elements.resultsContent.innerHTML = `
    <div>
      <h3>Akurasi pada 10.000 gambar uji</h3>
      <div class="result-row"><div class="result-label"><span>ViT kecil</span><strong>${accuracy(vit.test.accuracy)}</strong></div><div class="result-track"><div class="result-fill" style="width:${vit.test.accuracy * 100}%"></div></div></div>
      <div class="result-row cnn"><div class="result-label"><span>CNN kecil</span><strong>${accuracy(cnn.test.accuracy)}</strong></div><div class="result-track"><div class="result-fill" style="width:${cnn.test.accuracy * 100}%"></div></div></div>
      <p class="result-detail">Satu run per model, seed ${vit.seed}. Grafik ini menunjukkan hasil, bukan bukti bahwa satu arsitektur selalu lebih unggul.</p>
    </div>
    <div>
      <h3>Biaya eksperimen</h3>
      <div class="result-meta"><div><strong>${formatNumber(vit.parameters)}</strong><span>parameter ViT</span></div><div><strong>${formatNumber(cnn.parameters)}</strong><span>parameter CNN</span></div><div><strong>${vit.best_epoch}</strong><span>epoch terbaik ViT</span></div><div><strong>${cnn.best_epoch}</strong><span>epoch terbaik CNN</span></div><div><strong>${formatMinutes(vit.training_seconds)}</strong><span>waktu latihan ViT</span></div><div><strong>${formatMinutes(cnn.training_seconds)}</strong><span>waktu latihan CNN</span></div></div>
    </div>
  `;
  renderLossChart(vit.history, cnn.history);
  document.querySelector('#loss-section').hidden = false;
}

function renderLossChart(vitHistory, cnnHistory) {
  const width = 700;
  const height = 240;
  const left = 45;
  const right = 20;
  const top = 15;
  const bottom = 35;
  const maxEpoch = Math.max(vitHistory.at(-1).epoch, cnnHistory.at(-1).epoch);
  const maxLoss = Math.max(...vitHistory.map((row) => row.validation_loss), ...cnnHistory.map((row) => row.validation_loss)) * 1.1;
  const x = (epoch) => left + (epoch - 1) / Math.max(1, maxEpoch - 1) * (width - left - right);
  const y = (loss) => height - bottom - loss / maxLoss * (height - top - bottom);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Grafik loss validasi per epoch untuk ViT dan CNN kecil');
  const make = (tag, attributes) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
    svg.append(node);
    return node;
  };
  for (let step = 0; step <= 2; step += 1) {
    const value = maxLoss * step / 2;
    const point = y(value);
    make('line', { x1: left, y1: point, x2: width - right, y2: point, stroke: '#c9cec6', 'stroke-width': 1 });
    const label = make('text', { x: 0, y: point + 4, fill: '#3b4845', 'font-size': 12 });
    label.textContent = value.toFixed(2);
  }
  for (const epoch of [1, Math.ceil(maxEpoch / 2), maxEpoch]) {
    const label = make('text', { x: x(epoch), y: height - 8, fill: '#3b4845', 'font-size': 12, 'text-anchor': 'middle' });
    label.textContent = String(epoch);
  }
  for (const [history, color] of [[vitHistory, '#a23e20'], [cnnHistory, '#344c49']]) {
    const points = history.map((row) => `${x(row.epoch)},${y(row.validation_loss)}`).join(' ');
    make('polyline', { points, fill: 'none', stroke: color, 'stroke-width': 3, 'stroke-linejoin': 'round' });
    history.forEach((row) => make('circle', { cx: x(row.epoch), cy: y(row.validation_loss), r: 3.5, fill: color }));
  }
  elements.lossChart.replaceChildren(svg);
}

function renderStage(index) {
  const stage = stages[index];
  document.querySelector('#stage-index').textContent = `TAHAP 0${index + 1} / 04`;
  document.querySelector('#stage-formula').textContent = stage.formula;
  document.querySelector('#stage-title').textContent = stage.title;
  document.querySelector('#stage-description').textContent = stage.description;
  document.querySelector('#stage-code').textContent = stage.code;
  document.querySelectorAll('[data-stage]').forEach((button) => {
    const active = Number(button.dataset.stage) === index;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

async function loadDemo() {
  elements.loadState.hidden = false;
  elements.emptyState.hidden = true;
  elements.errorState.hidden = true;
  elements.labContent.hidden = true;
  document.querySelector('#loss-section').hidden = true;
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}data/demo.json`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.schema_version === 1 && Array.isArray(data.samples) && data.samples.length === 0) {
      elements.loadState.hidden = true;
      elements.emptyState.hidden = false;
      elements.resultsContent.textContent = 'Hasil belum tersedia. Jalankan ekspor model untuk mengisi data demo.';
      return;
    }
    if (data.schema_version !== 1 || data.samples.length !== 10) throw new Error('Invalid demo data');
    state.data = data;
    state.layer = data.config.depth - 1;
    state.head = 0;
    state.query = 0;
    addOptions(elements.querySelect, ['Token kelas [CLS]', ...Array.from({ length: 49 }, (_, index) => `Patch ${index + 1}`)]);
    addOptions(elements.layerSelect, Array.from({ length: data.config.depth }, (_, index) => `Lapis ${index + 1}`));
    addOptions(elements.headSelect, Array.from({ length: data.config.heads }, (_, index) => `Head ${index + 1}`));
    renderSamples();
    renderSample();
    renderResults();
    elements.loadState.hidden = true;
    elements.labContent.hidden = false;
  } catch (error) {
    console.error(error);
    elements.loadState.hidden = true;
    elements.errorState.hidden = false;
    elements.resultsContent.textContent = 'Hasil belum tersedia. Periksa berkas data demo dan coba lagi.';
  }
}

document.querySelectorAll('[data-stage]').forEach((button) => {
  button.addEventListener('click', () => renderStage(Number(button.dataset.stage)));
});
elements.querySelect.addEventListener('change', (event) => {
  state.query = Number(event.target.value);
  renderSample();
});
elements.layerSelect.addEventListener('change', (event) => {
  state.layer = Number(event.target.value);
  renderAttention();
});
elements.headSelect.addEventListener('change', (event) => {
  state.head = Number(event.target.value);
  renderAttention();
});
elements.sampleCanvas.addEventListener('click', (event) => {
  const bounds = elements.sampleCanvas.getBoundingClientRect();
  const column = Math.min(6, Math.floor((event.clientX - bounds.left) / bounds.width * 7));
  const row = Math.min(6, Math.floor((event.clientY - bounds.top) / bounds.height * 7));
  state.query = row * 7 + column + 1;
  renderSample();
});
document.querySelector('#retry-button').addEventListener('click', loadDemo);
document.querySelector('#empty-retry-button').addEventListener('click', loadDemo);
drawPlaceholder();
loadDemo();
