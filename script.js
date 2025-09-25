// ---------- Utilities ----------
function clamp(num, min, max) { return Math.min(max, Math.max(min, num)); }
function pct(x) { return typeof x === 'number' ? (x * 100).toFixed(1) + '%' : 'n/a'; }
function nowIso() { return new Date().toISOString(); }

// ---------- Mock data + analysis (parity with Python) ----------
function fetchRealtimeQuote(ticker) {
  const base = { AAPL: 225.0, MSFT: 435.0, GOOG: 165.0 };
  const basePrice = base[ticker] || 100.0;
  const price = basePrice * (1 + (Math.random() * 0.02 - 0.01));
  return {
    price: Number(price.toFixed(2)),
    change_pct: Number((Math.random() * 4 - 2).toFixed(2)),
    volume: Math.floor(Math.random() * (20000000 - 1000000) + 1000000)
  };
}

function fetchQuarterlies(ticker) {
  const mock = {
    AAPL: [
      { quarter_end: '2025-06-29', revenue: 87000000000, operating_income: 24000000000, net_income: 20000000000, eps: 1.43, eps_consensus: 1.38, guidance_revenue: 90000000000 },
      { quarter_end: '2025-03-30', revenue: 82000000000, operating_income: 22000000000, net_income: 18500000000, eps: 1.36, eps_consensus: 1.31, guidance_revenue: 85000000000 },
      { quarter_end: '2024-12-29', revenue: 91000000000, operating_income: 26000000000, net_income: 21000000000, eps: 1.50, eps_consensus: 1.47, guidance_revenue: 88000000000 },
      { quarter_end: '2024-09-29', revenue: 76000000000, operating_income: 19000000000, net_income: 16000000000, eps: 1.12, eps_consensus: 1.10, guidance_revenue: 80000000000 },
    ],
    MSFT: [
      { quarter_end: '2025-06-30', revenue: 68000000000, operating_income: 29000000000, net_income: 24000000000, eps: 2.95, eps_consensus: 2.85, guidance_revenue: 70000000000 },
      { quarter_end: '2025-03-31', revenue: 67000000000, operating_income: 28000000000, net_income: 23500000000, eps: 2.88, eps_consensus: 2.80, guidance_revenue: 69000000000 },
      { quarter_end: '2024-12-31', revenue: 65000000000, operating_income: 27000000000, net_income: 22800000000, eps: 2.76, eps_consensus: 2.70, guidance_revenue: 66000000000 },
      { quarter_end: '2024-09-30', revenue: 62000000000, operating_income: 25000000000, net_income: 21000000000, eps: 2.55, eps_consensus: 2.50, guidance_revenue: 64000000000 },
    ],
  };
  return mock[ticker] || mock['AAPL'];
}

function computeGrowth(curr, prev) { if (!prev) return null; return (curr - prev) / prev; }
function buildQuarterlyContext(quarters) {
  if (!quarters || quarters.length === 0) return { latest_quarter: null, qoq_growth: {}, yoy_growth: {}, eps_surprise_pct: null, guidance_delta_pct: null, trend_notes: [] };
  const latest = quarters[0];
  const prevQ = quarters[1];
  const prevY = quarters[3];
  const qoq = {
    revenue: prevQ ? computeGrowth(latest.revenue, prevQ.revenue) : null,
    operating_income: prevQ ? computeGrowth(latest.operating_income, prevQ.operating_income) : null,
    net_income: prevQ ? computeGrowth(latest.net_income, prevQ.net_income) : null,
    eps: prevQ ? computeGrowth(latest.eps, prevQ.eps) : null,
  };
  const yoy = {
    revenue: prevY ? computeGrowth(latest.revenue, prevY.revenue) : null,
    operating_income: prevY ? computeGrowth(latest.operating_income, prevY.operating_income) : null,
    net_income: prevY ? computeGrowth(latest.net_income, prevY.net_income) : null,
    eps: prevY ? computeGrowth(latest.eps, prevY.eps) : null,
  };
  let eps_surprise_pct = null;
  if (typeof latest.eps === 'number' && typeof latest.eps_consensus === 'number') {
    eps_surprise_pct = computeGrowth(latest.eps, latest.eps_consensus);
  }
  let guidance_delta_pct = null;
  if (typeof latest.guidance_revenue === 'number' && prevQ && typeof prevQ.revenue === 'number') {
    guidance_delta_pct = computeGrowth(latest.guidance_revenue, prevQ.revenue);
  }
  const notes = [];
  if (typeof qoq.revenue === 'number') { notes.push(`${qoq.revenue > 0 ? 'QoQ revenue accelerating' : 'QoQ revenue contracting'} (${pct(qoq.revenue)}).`); }
  if (typeof eps_surprise_pct === 'number') { notes.push(`EPS surprise ${pct(eps_surprise_pct)} vs consensus.`); }
  if (typeof guidance_delta_pct === 'number') { notes.push(`Next guidance ${guidance_delta_pct > 0 ? 'above' : 'below'} last quarter revenue by ${pct(guidance_delta_pct)}.`); }
  return { latest_quarter: latest, qoq_growth: qoq, yoy_growth: yoy, eps_surprise_pct, guidance_delta_pct, trend_notes: notes };
}

function evaluateFinancialHealth(ticker) {
  const mock = {
    AAPL: { pe_ratio: 28.5, revenue_growth: 0.15, debt_to_equity: 1.2 },
    MSFT: { pe_ratio: 35.0, revenue_growth: 0.20, debt_to_equity: 1.0 },
    GOOG: { pe_ratio: 25.0, revenue_growth: 0.18, debt_to_equity: 0.8 },
  };
  let fin = mock[ticker];
  if (!fin) {
    fin = { pe_ratio: 15 + Math.random() * 35, revenue_growth: 0.05 + Math.random() * 0.25, debt_to_equity: 0.5 + Math.random() * 1.5 };
  }
  let pe_score = fin.pe_ratio > 15 ? 100 - (fin.pe_ratio - 15) * 2 : 100;
  let revenue_score = Math.min(100, fin.revenue_growth * 500);
  let debt_score = 100 - (fin.debt_to_equity - 0.5) * 40;
  pe_score = clamp(pe_score, 0, 100);
  revenue_score = clamp(revenue_score, 0, 100);
  debt_score = clamp(debt_score, 0, 100);
  const base_score = Math.round(pe_score * 0.4 + revenue_score * 0.4 + debt_score * 0.2);
  const quarterly = fetchQuarterlies(ticker);
  const qctx = buildQuarterlyContext(quarterly);
  const quote = fetchRealtimeQuote(ticker);
  const series = (arr) => arr.filter(v => typeof v === 'number');
  const rev_series = series(quarterly.map(q => q && q.revenue));
  const op_series = series(quarterly.map(q => q && q.operating_income));
  const ni_series = series(quarterly.map(q => q && q.net_income));
  const ttm = (vals) => vals.length >= 4 ? vals.slice(0,4).reduce((a,b)=>a+b,0) : null;
  const ttm_rev = ttm(rev_series);
  const ttm_op = ttm(op_series);
  const ttm_ni = ttm(ni_series);
  const ttm_op_margin = ttm_rev && ttm_op != null ? ttm_op / ttm_rev : null;
  const ttm_net_margin = ttm_rev && ttm_ni != null ? ttm_ni / ttm_rev : null;
  const qoq_growth_series = [];
  for (let i=0; i<Math.min(rev_series.length-1,7); i++) {
    const prev = rev_series[i+1]; const curr = rev_series[i]; if (prev) qoq_growth_series.push((curr-prev)/prev);
  }
  let growth_volatility = null;
  if (qoq_growth_series.length) {
    const mean = qoq_growth_series.reduce((a,b)=>a+b,0)/qoq_growth_series.length;
    const variance = qoq_growth_series.reduce((a,b)=>a+(b-mean)**2,0)/qoq_growth_series.length;
    growth_volatility = Math.sqrt(variance);
  }
  let margin_trend = null;
  if (rev_series.length>=2 && op_series.length>=2) {
    const margins = [];
    for (let i=0;i<Math.min(rev_series.length, op_series.length);i++) { const r=rev_series[i], o=op_series[i]; if (r) margins.push(o/r); }
    if (margins.length>=2) { margin_trend = (margins[0] - margins[margins.length-1]) / Math.max(1, margins.length-1); }
  }
  const yoy_rev = qctx.yoy_growth && typeof qctx.yoy_growth.revenue === 'number' ? qctx.yoy_growth.revenue : null;
  const expected_growth = typeof yoy_rev === 'number' ? Math.max(0.01, Math.min(0.50, yoy_rev)) : null;
  const peg_like = expected_growth && fin.pe_ratio ? fin.pe_ratio / (expected_growth * 100) : null;
  const components = {};
  if (typeof ttm_op_margin === 'number') components.margin_quality = clamp(30 + (ttm_op_margin * 175), 0, 100);
  if (typeof margin_trend === 'number') components.margin_trend = clamp(50 + (margin_trend * 1000), 0, 100);
  if (typeof growth_volatility === 'number') components.growth_stability = clamp(100 - (growth_volatility * 300), 0, 100);
  if (typeof peg_like === 'number') components.valuation_vs_growth = clamp(110 - (peg_like * 40), 0, 100);
  const rigorous_avg = Object.keys(components).length ? Object.values(components).reduce((a,b)=>a+b,0)/Object.keys(components).length : base_score;
  const final_score = Math.round(base_score * 0.5 + rigorous_avg * 0.5);
  const notes = [];
  if (typeof quote.change_pct === 'number') {
    if (quote.change_pct > 0) notes.push(`Shares are up ${quote.change_pct.toFixed(2)}% today; momentum supports fundamentals.`);
    else if (quote.change_pct < 0) notes.push(`Shares are down ${Math.abs(quote.change_pct).toFixed(2)}% today; watch for risk-off sentiment.`);
  }
  if (qctx.trend_notes && qctx.trend_notes.length) notes.push(...qctx.trend_notes);
  notes.push(`Composite health score at ${final_score}/100 based on valuation, growth, leverage.`);
  return {
    as_of: nowIso(), ticker, score: final_score,
    metrics: { ...fin, valuation_comment: 'Lower P/E is generally better; growth can justify higher multiples.', leverage_comment: 'Lower debt-to-equity indicates less balance sheet risk.' },
    quarterly_context: qctx, realtime_quote: quote,
    rigorous_context: { ttm_revenue: ttm_rev, ttm_operating_income: ttm_op, ttm_net_income: ttm_ni, ttm_operating_margin: ttm_op_margin, ttm_net_margin: ttm_net_margin, growth_volatility, margin_trend_per_quarter: margin_trend, peg_like, components, base_score, final_score },
    narrative_notes: notes,
  };
}

// Sentiment + Risk
function fakeHeadlines(ticker) {
  const now = new Date();
  function iso(d) { return d.toISOString(); }
  return [
    { title: `Positive Q3 earnings report for ${ticker} drives stock price up.`, published_at: iso(new Date(now.getTime()-35*60000)), source: 'MockWire' },
    { title: `New product launch from ${ticker} receives mixed reviews from tech experts.`, published_at: iso(new Date(now.getTime()-2*3600000)), source: 'TechMock' },
    { title: `Analysts downgrade ${ticker} due to supply chain concerns.`, published_at: iso(new Date(now.getTime()-6*3600000)), source: 'StreetMock' },
    { title: `Innovative R&D team at ${ticker} wins prestigious award.`, published_at: iso(new Date(now.getTime()-24*3600000)), source: 'AwardsDaily' },
    { title: `${ticker} announces a new partnership to expand into emerging markets.`, published_at: iso(new Date(now.getTime()-27*3600000)), source: 'BizMock' },
    { title: `A brief overview of ${ticker}'s performance this quarter.`, published_at: iso(new Date(now.getTime()-48*3600000)), source: 'MarketDaily' },
  ];
}

function polarity(text) {
  // Lightweight heuristic replacement for TextBlob polarity (-1..1)
  const t = text.toLowerCase();
  const pos = ['up','award','positive','wins','expand','partnership','beat','strong','growth'];
  const neg = ['down','concerns','downgrade','miss','light','scrutiny','outage','breach','recall','layoff'];
  let score = 0;
  pos.forEach(k=>{ if (t.includes(k)) score += 0.2; });
  neg.forEach(k=>{ if (t.includes(k)) score -= 0.25; });
  return Math.max(-1, Math.min(1, score));
}

function classifyContext(title) {
  const t = title.toLowerCase();
  if (t.includes('earnings') || (t.includes('q') && t.includes('report'))) return 'earnings';
  if (t.includes('guidance')) return 'guidance';
  if (t.includes('downgrade') || t.includes('upgrade')) return 'analyst_rating';
  if (t.includes('launch') || t.includes('partnership')) return 'strategic';
  return 'general';
}

async function fetchExternalHeadlines(ticker) {
  if (!window.__settings || !window.__settings.newsUrl) return null;
  const url = window.__settings.newsUrl.replace('{ticker}', encodeURIComponent(ticker));
  try {
    const res = await fetch(url, { cache: 'no-store' });
    const data = await res.json();
    const arr = Array.isArray(data) ? data : (Array.isArray(data.headlines) ? data.headlines : []);
    return arr.map(h => {
      if (typeof h === 'string') return { title: h, published_at: nowIso(), source: 'external' };
      return { title: h.title || '', published_at: h.published_at || nowIso(), source: h.source || 'external' };
    });
  } catch (e) {
    appendBot('Failed to fetch external headlines; using mock.');
    return null;
  }
}

function computeSentimentFromHeadlines(items) {
  const enriched = [];
  const sentiments = [];
  items.forEach(it => {
    const pol = polarity(it.title);
    sentiments.push(pol);
    enriched.push({ title: it.title, published_at: it.published_at, source: it.source, polarity: pol, context: classifyContext(it.title) });
  });
  const avg = sentiments.length ? sentiments.reduce((a,b)=>a+b,0) / sentiments.length : 0;
  const buckets = { earnings:[], guidance:[], analyst_rating:[], strategic:[], general:[] };
  enriched.forEach(h => buckets[h.context].push(h));
  const notes = [];
  if (buckets.earnings.length) notes.push('Earnings-related coverage present; consider EPS surprise in fundamentals.');
  if (buckets.guidance.length) notes.push('Guidance-focused headlines may drive forward-looking sentiment.');
  if (buckets.analyst_rating.length) notes.push('Analyst rating changes detected; short-term volatility likely.');
  if (buckets.strategic.length) notes.push('Strategic news (launches/partnerships) could shift growth narrative.');
  return { average_score: avg, headlines: enriched, narrative_notes: notes };
}

async function analyzeMediaSentiment(ticker) {
  const external = await fetchExternalHeadlines(ticker);
  const items = external && external.length ? external : fakeHeadlines(ticker);
  const base = computeSentimentFromHeadlines(items);
  return { as_of: nowIso(), ticker, ...base };
}

function fakeSocial(ticker) {
  const now = Date.now();
  function mk(text, minutes, platform, likes, replies, shares, followers) {
    return { text, published_at: new Date(now - minutes*60000).toISOString(), platform, engagement: { likes, replies, shares }, author_followers: followers };
  }
  return [
    mk(`${ticker} earnings beat but guidance light; sell the rip?`, 20, 'twitter', 320, 95, 110, 50000),
    mk(`Big ${ticker} outage reported by users across EU region.`, 45, 'reddit', 210, 140, 60, 15000),
    mk(`Hearing ${ticker} faces potential regulatory scrutiny over acquisition.`, 75, 'twitter', 410, 180, 200, 120000),
    mk(`Loving the new ${ticker} feature rollout. Smooth UX.`, 130, 'twitter', 85, 12, 9, 8000),
    mk(`Rumors of layoffs at ${ticker} next quarter.`, 200, 'stocktwits', 150, 60, 30, 10000),
  ];
}

async function fetchExternalSocial(ticker) {
  if (!window.__settings || !window.__settings.socialUrl) return null;
  const url = window.__settings.socialUrl.replace('{ticker}', encodeURIComponent(ticker));
  try {
    const res = await fetch(url, { cache: 'no-store' });
    const data = await res.json();
    const arr = Array.isArray(data) ? data : (Array.isArray(data.posts) ? data.posts : []);
    return arr.map(p => {
      if (typeof p === 'string') return { text: p, published_at: nowIso(), platform: 'external', engagement: { likes:0, replies:0, shares:0 }, author_followers: 0 };
      return {
        text: p.text || '',
        published_at: p.published_at || nowIso(),
        platform: p.platform || 'external',
        engagement: p.engagement || { likes:0, replies:0, shares:0 },
        author_followers: p.author_followers || 0,
      };
    });
  } catch (e) {
    appendBot('Failed to fetch external social posts; using mock.');
    return null;
  }
}

function containsTopics(text, topics) { const t = text.toLowerCase(); return topics.some(k => t.includes(k)); }
function socialVolumeIndex(posts) {
  if (!posts.length) return 0;
  const countScore = posts.length / 10.0;
  const engagementScore = posts.reduce((a,p)=>a + (p.engagement.likes + p.engagement.replies + p.engagement.shares), 0) / 1000.0;
  const reachScore = posts.reduce((a,p)=>a + (p.author_followers||0), 0) / 500000.0;
  return Math.max(0, countScore + engagementScore + reachScore);
}

async function analyzeRiskFactors(ticker, media) {
  const ext = await fetchExternalSocial(ticker);
  const posts = ext && ext.length ? ext : fakeSocial(ticker);
  const socialScores = [];
  const controversyKeywords = ['lawsuit','probe','scrutiny','regulator','antitrust','outage','breach','recall','layoff','guidance light','miss'];
  const controversyFlags = [];
  posts.forEach(p => {
    const pol = polarity(p.text);
    socialScores.push(pol);
    if (containsTopics(p.text, controversyKeywords)) {
      for (const k of controversyKeywords) { if (p.text.toLowerCase().includes(k)) { controversyFlags.push(k); break; } }
    }
  });
  const socialAvg = socialScores.length ? socialScores.reduce((a,b)=>a+b,0)/socialScores.length : 0;
  const socialNegRatio = socialScores.length ? socialScores.filter(s => s < 0).length / socialScores.length : 0;
  const sVol = socialVolumeIndex(posts);
  const mediaAvg = media.average_score || 0;
  const mediaNegRatio = media.headlines && media.headlines.length ? media.headlines.filter(h => (h.polarity||0) < 0).length / media.headlines.length : 0;
  const volumeComp = Math.min(1, sVol);
  const controversyComp = Math.min(1, new Set(controversyFlags).size / 3.0);
  const mediaNegComp = (mediaNegRatio * 0.7) + Math.max(0, -mediaAvg) * 0.3;
  const socialNegComp = (socialNegRatio * 0.6) + Math.max(0, -socialAvg) * 0.4;
  const composite = 0.35*mediaNegComp + 0.35*socialNegComp + 0.20*volumeComp + 0.10*controversyComp;
  const riskScore = Math.round(clamp(composite, 0, 1) * 100);
  const notes = [];
  if (riskScore >= 70) notes.push('Elevated risk driven by negative tone and/or chatter spike.');
  else if (riskScore >= 40) notes.push('Moderate risk; monitor sentiment momentum and topics.');
  else notes.push('Low immediate sentiment-driven risk, barring new headlines.');
  if (volumeComp > 0.6) notes.push('Social volume spike suggests higher short-term volatility.');
  if (controversyFlags.length) notes.push('Controversy topics detected: ' + Array.from(new Set(controversyFlags)).sort().join(', ') + '.');
  if (media.narrative_notes && media.narrative_notes.length) notes.push(...media.narrative_notes);
  return { as_of: nowIso(), ticker, risk_score: riskScore, components: { media_negative_ratio: mediaNegRatio, media_avg: mediaAvg, social_negative_ratio: socialNegRatio, social_avg: socialAvg, social_volume_index: sVol, controversy_intensity: controversyComp }, media, social_posts: posts, topic_flags: Array.from(new Set(controversyFlags)).sort(), narrative_notes: notes };
}

// ---------- UI ----------
const chatbox = document.getElementById('chatbox');
const input = document.getElementById('userInput');
const btn = document.getElementById('sendBtn');
const qaInput = document.getElementById('qaInput');
const askBtn = document.getElementById('askBtn');

function appendUser(text) { chatbox.innerHTML += `<div class="user">You: ${text}</div>`; }
function appendBot(html) { chatbox.innerHTML += `<div class="bot">${html}</div>`; chatbox.scrollTop = chatbox.scrollHeight; }

let lastResult = null;
function renderResponse(fin, sent, risk) {
  const finMetrics = fin.metrics || {};
  const headlinesCount = (sent.headlines && sent.headlines.length) || 0;
  const flags = (risk.topic_flags || []).join(', ') || 'None';
  const notes = [...(fin.narrative_notes||[]), ...(sent.narrative_notes||[]), ...(risk.narrative_notes||[])];
  const notesHtml = notes.length ? `<ul class="notes">${notes.map(n=>`<li>${n}</li>`).join('')}</ul>` : '';
  lastResult = { financial: fin, sentiment: sent, risk };
  return `
    <div><strong>📊 Financial Health for ${fin.ticker}:</strong> Score ${fin.score}/100</div>
    <div class="metrics">
      <div class="card">PE Ratio: ${finMetrics.pe_ratio?.toFixed ? finMetrics.pe_ratio.toFixed(2) : finMetrics.pe_ratio}</div>
      <div class="card">Revenue Growth: ${pct(finMetrics.revenue_growth)}</div>
      <div class="card">Debt/Equity: ${finMetrics.debt_to_equity?.toFixed ? finMetrics.debt_to_equity.toFixed(2) : finMetrics.debt_to_equity}</div>
      <div class="card">Price: ${fin.realtime_quote?.price ?? 'n/a'} (${fin.realtime_quote?.change_pct ?? 'n/a'}%)</div>
    </div>
    <div class="hr"></div>
    <div><strong>📰 Media Sentiment:</strong> Avg score ${sent.average_score.toFixed(2)}</div>
    <div class="meta">Headlines analyzed: ${headlinesCount}</div>
    <div class="hr"></div>
    <div><strong>⚠️ Risk Score:</strong> ${risk.risk_score} / 100</div>
    <div class="meta">Key flags: ${flags}</div>
    ${notesHtml}
    <div class="footer">As of ${fin.as_of}</div>
  `;
}

function validateTicker(t) {
  const s = String(t||'').trim().toUpperCase();
  if (!s) return { ok:false, error: 'Please enter a stock ticker symbol.' };
  if (!(s.length >= 1 && s.length <= 10)) return { ok:false, error:'Invalid ticker format. Use letters/numbers (e.g., AAPL, MSFT).' };
  for (const ch of s) { if (!(/[A-Z0-9.-]/.test(ch))) return { ok:false, error:'Invalid ticker format. Use letters/numbers (e.g., AAPL, MSFT).' }; }
  return { ok:true, value:s };
}

function setHash(ticker) { try { location.hash = encodeURIComponent(ticker); } catch(e){} }
function getHashTicker() { try { return decodeURIComponent((location.hash||'').replace(/^#/, '')); } catch(e){ return ''; } }

let liveTimer = null;
function clearLive() { if (liveTimer) { clearInterval(liveTimer); liveTimer = null; } }

async function runAnalysisOnce(ticker) {
  const fin = evaluateFinancialHealth(ticker);
  const sent = await analyzeMediaSentiment(ticker);
  const risk = await analyzeRiskFactors(ticker, sent);
  const html = renderResponse(fin, sent, risk);
  appendBot(html);
  try {
    const auto = ragAnswerFromContext({ question: `What are the key positives and risks for ${ticker} right now?`, sentiment: sent, risk });
    if (auto && auto.answerHtml) appendBot(auto.answerHtml);
  } catch (e) { /* noop */ }
}

async function analyze() {
  const raw = input.value; const v = validateTicker(raw);
  if (!v.ok) { appendBot(v.error); return; }
  const ticker = v.value;
  appendUser(ticker);
  btn.disabled = true;
  try {
    await runAnalysisOnce(ticker);
    setHash(ticker);
  } catch (e) {
    appendBot(`Error analyzing ${ticker}: ${e?.message || e}`);
  } finally {
    btn.disabled = false; input.value = ''; input.focus();
  }
}

btn.addEventListener('click', analyze);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') analyze(); });

// ---------- Lightweight client-side RAG (enhanced) ----------
// Chunking, embeddings (worker-hashed) with TF-IDF fallback, IndexedDB-backed vector index, and LRU per ticker.

const RAG = {
  indexByTicker: new Map(), // ticker -> { chunks: Array<Chunk>, vectors: Float32Array[], ids: string[] }
  lruByTicker: new Map(),   // ticker -> Set of ids (maintain recency)
  idb: null,
  worker: null,
  embeddingsReady: false,
  lastBuiltFor: null,
};

function initEmbeddingsWorker() {
  if (RAG.worker) return;
  const workerCode = `
    let dim = 384;
    function hash32(str){ let h=2166136261>>>0; for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=(h*16777619)>>>0; } return h>>>0; }
    function tokenVec(token){ const h = hash32(token); const v = new Float32Array(dim); let seed=h; for(let i=0;i<dim;i++){ seed = (1664525*seed + 1013904223)>>>0; const x = ((seed>>>9)&0x7fffff)/0x7fffff; v[i] = (x*2-1); } return v; }
    function normalize(v){ let s=0; for(let i=0;i<v.length;i++){ s+=v[i]*v[i]; } s=Math.sqrt(s)||1; for(let i=0;i<v.length;i++){ v[i]/=s; } }
    function embedOne(text){ const toks = String(text||'').toLowerCase().replace(/[^a-z0-9\s]/g,' ').split(/\s+/).filter(t=>t&&t.length>1); const vec = new Float32Array(dim); for(const t of toks){ const tv = tokenVec(t); for(let i=0;i<dim;i++){ vec[i]+=tv[i]; } } normalize(vec); return vec; }
    onmessage = async (e)=>{
      const { id, type, payload } = e.data||{};
      if (type === 'embed'){
        try {
          const arr = payload.texts||[];
          const out = arr.map(txt=>embedOne(txt));
          postMessage({ id, ok:true, vectors: out });
        } catch (err) {
          postMessage({ id, ok:false, error: String(err&&err.message||err) });
        }
      } else if (type==='dim'){ dim = payload?.dim||dim; postMessage({ id, ok:true, dim }); }
    };
  `;
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  try {
    const url = URL.createObjectURL(blob);
    const w = new Worker(url);
    RAG.worker = w;
    RAG.embeddingsReady = true;
  } catch (e) {
    RAG.worker = null;
    RAG.embeddingsReady = false;
  }
}

function workerEmbed(texts) {
  return new Promise((resolve) => {
    if (!RAG.worker || !RAG.embeddingsReady) { resolve(null); return; }
    const id = Math.random().toString(36).slice(2);
    const onMsg = (ev) => {
      if (ev.data && ev.data.id === id) {
        RAG.worker.removeEventListener('message', onMsg);
        if (ev.data.ok) resolve(ev.data.vectors);
        else resolve(null);
      }
    };
    RAG.worker.addEventListener('message', onMsg);
    RAG.worker.postMessage({ id, type: 'embed', payload: { texts } });
  });
}

// IndexedDB minimal wrapper
function openVectorDB() {
  return new Promise((resolve) => {
    if (RAG.idb) { resolve(RAG.idb); return; }
    try {
      const req = indexedDB.open('rag_index_db', 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('chunks')) {
          db.createObjectStore('chunks', { keyPath: 'key' });
        }
      };
      req.onsuccess = () => { RAG.idb = req.result; resolve(RAG.idb); };
      req.onerror = () => { resolve(null); };
    } catch (e) { resolve(null); }
  });
}

async function idbSaveChunks(ticker, records) {
  const db = await openVectorDB(); if (!db) return;
  const tx = db.transaction('chunks', 'readwrite');
  const store = tx.objectStore('chunks');
  for (const rec of records) { store.put({ key: `${ticker}:${rec.id}`, ticker, rec }); }
}

async function idbLoadChunks(ticker) {
  const db = await openVectorDB(); if (!db) return [];
  return new Promise((resolve) => {
    const tx = db.transaction('chunks', 'readonly');
    const store = tx.objectStore('chunks');
    const out = [];
    const req = store.openCursor();
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        const v = cursor.value;
        if (v && v.ticker === ticker) out.push(v.rec);
        cursor.continue();
      } else {
        resolve(out);
      }
    };
    req.onerror = () => resolve([]);
  });
}

// Chunking
function chunkText(text, opts) {
  const tokens = tokenize(text);
  const maxTokens = clamp((opts&&opts.maxTokens)||320, 64, 512);
  const overlap = clamp((opts&&opts.overlap)||80, 0, Math.floor(maxTokens/2));
  const chunks = [];
  let start = 0; let idx = 0;
  while (start < tokens.length) {
    const windowTokens = tokens.slice(start, start + maxTokens);
    const chunk = windowTokens.join(' ');
    if (chunk.trim()) chunks.push({ text: chunk, idx });
    if (start + maxTokens >= tokens.length) break;
    start += (maxTokens - overlap);
    idx++;
  }
  return chunks;
}

function buildCorpusFromResults(ticker, sentiment, risk) {
  const docs = [];
  const headlines = (sentiment && sentiment.headlines) || [];
  headlines.forEach((h, i) => {
    const baseId = `H${i+1}`;
    const chunks = chunkText(h.title||'', { maxTokens: 320, overlap: 80 });
    chunks.forEach((c) => {
      docs.push({
        id: `${baseId}#${c.idx}`,
        text: c.text,
        meta: { ticker, type: 'headline', source: h.source, published_at: h.published_at, title: h.title, context: h.context }
      });
    });
  });
  const posts = (risk && risk.social_posts) || [];
  posts.forEach((p, i) => {
    const baseId = `S${i+1}`;
    const chunks = chunkText(p.text||'', { maxTokens: 320, overlap: 80 });
    chunks.forEach((c) => {
      docs.push({
        id: `${baseId}#${c.idx}`,
        text: c.text,
        meta: { ticker, type: 'social', platform: p.platform, published_at: p.published_at, title: p.text.slice(0,140), context: 'social' }
      });
    });
  });
  return docs;
}

function ensureLRU(ticker) {
  if (!RAG.lruByTicker.has(ticker)) RAG.lruByTicker.set(ticker, new Map());
  return RAG.lruByTicker.get(ticker);
}

function touchLRU(ticker, id, payload) {
  const lru = ensureLRU(ticker);
  if (lru.has(id)) lru.delete(id);
  lru.set(id, payload);
  while (lru.size > 1000) {
    const oldest = lru.keys().next().value;
    lru.delete(oldest);
  }
}

async function rebuildVectorIndex(ticker, sentiment, risk) {
  initEmbeddingsWorker();
  const corpus = buildCorpusFromResults(ticker, sentiment, risk);
  // LRU enforce
  const lruMap = ensureLRU(ticker);
  corpus.forEach(d => touchLRU(ticker, d.id, d));
  const limited = Array.from(lruMap.values());
  // Embed or fallback
  let vectors = null;
  if (RAG.embeddingsReady) {
    vectors = await workerEmbed(limited.map(d=>d.text));
  }
  const bundle = { chunks: limited, vectors, ids: limited.map(d=>d.id) };
  RAG.indexByTicker.set(ticker, bundle);
  // Persist
  const records = limited.map((d, i) => ({ id: d.id, text: d.text, meta: d.meta, vec: vectors && vectors[i] ? Array.from(vectors[i]) : null }));
  idbSaveChunks(ticker, records);
  RAG.lastBuiltFor = ticker;
  return { count: limited.length };
}

function cosineVec(a, b) {
  let s=0, na=0, nb=0; const n=a.length; for(let i=0;i<n;i++){ const x=a[i], y=b[i]; s+=x*y; na+=x*x; nb+=y*y; } return s/((Math.sqrt(na)||1)*(Math.sqrt(nb)||1));
}

async function rankDocsEmbeddings(question, ticker, k=5) {
  const bundle = RAG.indexByTicker.get(ticker);
  if (!bundle || !bundle.chunks || !bundle.chunks.length) return null;
  if (!bundle.vectors || !RAG.embeddingsReady) return null;
  const qv = await workerEmbed([question]).then(v=>v&&v[0]||null);
  if (!qv) return null;
  const scores = bundle.vectors.map((vec, i) => ({ doc: bundle.chunks[i], score: cosineVec(qv, vec) }));
  scores.sort((a,b)=>b.score - a.score);
  return scores.slice(0, k);
}
function tokenize(text) {
  return String(text||'')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t && t.length > 1);
}

function buildCorpus(sentiment, risk) { // legacy simple corpus (TF-IDF path)
  const docs = [];
  const headlines = (sentiment && sentiment.headlines) || [];
  headlines.forEach((h, i) => { docs.push({ id: `H${i+1}`, text: h.title||'', meta: { type:'headline', source:h.source, published_at:h.published_at } }); });
  const posts = (risk && risk.social_posts) || [];
  posts.forEach((p, i) => { docs.push({ id:`S${i+1}`, text: p.text||'', meta:{ type:'social', platform:p.platform, published_at:p.published_at } }); });
  return docs;
}

function computeTf(tokens) {
  const tf = Object.create(null);
  tokens.forEach(t => { tf[t] = (tf[t]||0) + 1; });
  const denom = tokens.length || 1;
  Object.keys(tf).forEach(k => { tf[k] = tf[k] / denom; });
  return tf;
}

function computeIdf(allDocsTokens) {
  const df = Object.create(null);
  allDocsTokens.forEach(tokens => {
    const uniq = Array.from(new Set(tokens));
    uniq.forEach(t => { df[t] = (df[t]||0) + 1; });
  });
  const N = allDocsTokens.length || 1;
  const idf = Object.create(null);
  Object.keys(df).forEach(t => { idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1; });
  return idf;
}

function dot(a, b) {
  let s = 0;
  const keys = a.size < b.size ? a : b;
  keys.forEach((v, k) => { if (b.has(k)) s += v * b.get(k); });
  return s;
}

function toWeightedVector(tf, idf) {
  const m = new Map();
  Object.keys(tf).forEach(t => { const w = (idf[t]||0) * tf[t]; if (w) m.set(t, w); });
  return m;
}

function l2(m) { let s=0; m.forEach(v=>{ s+=v*v; }); return Math.sqrt(s)||1; }

function cosineSim(a, b) { return dot(a,b) / (l2(a) * l2(b)); }

function rankDocs(question, docs, k=5) {
  const docTokens = docs.map(d => tokenize(d.text));
  const idf = computeIdf(docTokens);
  const docVecs = docTokens.map(tokens => toWeightedVector(computeTf(tokens), idf));
  const qVec = toWeightedVector(computeTf(tokenize(question)), idf);
  const scored = docs.map((d, i) => ({ doc: d, score: cosineSim(qVec, docVecs[i]) }));
  return scored.sort((a,b)=>b.score - a.score).slice(0, k);
}

function formatCitations(top) {
  return top.map((it, idx) => {
    const m = it.doc.meta||{};
    const tag = m.type === 'headline' ? (m.source || 'news') : (m.platform || 'social');
    const ts = m.published_at ? ` — ${new Date(m.published_at).toLocaleString()}` : '';
    return `[${idx+1}] (${tag}) ${it.doc.text}${ts}`;
  }).join('<br>');
}

function synthesizeAnswer(question, top) {
  // Simple heuristic summary from top snippets
  const positives = ['beat','up','award','wins','expand','partnership','strong','growth','launch'];
  const negatives = ['down','concerns','downgrade','miss','light','scrutiny','outage','breach','recall','layoff','probe','lawsuit'];
  const pos = []; const neg = [];
  top.forEach((it, idx) => {
    const t = (it.doc.text||'').toLowerCase();
    if (positives.some(k=>t.includes(k))) pos.push(idx+1);
    if (negatives.some(k=>t.includes(k))) neg.push(idx+1);
  });
  const bullets = [];
  if (pos.length) bullets.push(`Positives hinted in ${pos.map(n=>`[${n}]`).join(', ')}`);
  if (neg.length) bullets.push(`Risks flagged in ${neg.map(n=>`[${n}]`).join(', ')}`);
  const lead = bullets.length ? bullets.join(' · ') : 'Synthesized from top relevant items below.';
  const cited = top.map((it, idx) => `${it.doc.text} [${idx+1}]`).join(' ');
  const answer = `${lead}`;
  return { answer, cited };
}

async function ragAnswerFromContext({ question, sentiment, risk, k=5 }) {
  const ticker = lastResult?.financial?.ticker || sentiment?.ticker || risk?.ticker;
  if (!ticker) return { answerHtml: '<div class="bot">No ticker context.</div>' };
  // Try embeddings first
  let top = await rankDocsEmbeddings(question, ticker, k);
  let usedEmb = true;
  if (!top) {
    usedEmb = false;
    const docs = buildCorpus(sentiment, risk);
    if (!docs.length) return { answerHtml: '<div class="bot">No context available for RAG.</div>' };
    top = rankDocs(question, docs, k);
  }
  const { answer } = synthesizeAnswer(question, top);
  const citationsHtml = formatCitations(top);
  const why = usedEmb ? 'MiniLM-like embedding similarity (worker-hashed) + recency' : 'TF‑IDF cosine similarity';
  const html = `
    <div><strong>🔎 RAG Answer:</strong> ${answer}</div>
    <div class="meta">Question: ${question}</div>
    <div class="meta">Retriever: ${why}</div>
    <div class="hr"></div>
    <div class="meta"><strong>Sources</strong></div>
    <div class="meta">${citationsHtml}</div>
  `;
  return { answerHtml: html, top, usedEmb, answer };
}

function renderAnswerPanel(question, ranked, opts) {
  const box = document.getElementById('answerBox');
  const showSources = document.getElementById('toggleSources')?.checked;
  const top3 = ranked.slice(0,3);
  const bullets = [];
  // rationale bullets based on contexts
  const contexts = new Set();
  top3.forEach(it=>{ const c=it.doc?.meta?.context; if (c) contexts.add(c); });
  if (contexts.size) bullets.push(`Contexts: ${Array.from(contexts).join(', ')}`);
  const chips = (lastResult?.risk?.topic_flags||[]).map(t=>`<span style="display:inline-block; background:#e8f0fe; color:#174ea6; padding:2px 6px; border-radius:10px; margin-right:4px; font-size:11px;">${t}</span>`).join(' ');
  const ticker = lastResult?.financial?.ticker;
  const baseline = loadScoreBaseline(ticker);
  const items = top3.map((it, idx)=>{
    const m = it.doc.meta||{};
    const tag = m.type==='headline' ? (m.source||'news') : (m.platform||'social');
    const ts = m.published_at ? new Date(m.published_at).toLocaleString() : '';
    const conf = (it.score ?? 0).toFixed(3);
    const z = computeZ(it.score ?? 0, baseline);
    const body = showSources ? `<div class="meta" style="margin-top:4px;">${it.doc.text}</div>` : '';
    return `<div class="card"><div><strong>[${idx+1}] ${tag}</strong> · <span class="meta">${ts}</span> · <span class="meta">confidence ${conf}</span> · <span class="meta">z ${z}</span></div>${body}</div>`;
  }).join('');
  box.innerHTML = `
    <div style="margin-bottom:6px;">${chips}</div>
    ${items}
  `;
  const why = document.getElementById('whyAnswerBody');
  if (why) {
    const retriever = opts?.usedEmb ? 'Embedding similarity (worker)' : 'TF‑IDF cosine';
    why.innerHTML = `<ul class="notes"><li>Retriever: ${retriever}</li><li>Max-similarity over chunked windows (≈300 tokens)</li><li>Recency via latest headlines/social</li></ul>`;
  }
  if (ticker) updateScoreBaseline(ticker, ranked.map(r=>r.score||0));
}

// ---- Score normalization (per-ticker z-scores; 7-day proxy via session history) ----
function loadScoreBaseline(ticker) {
  if (!ticker) return { mean: 0, std: 1, n: 0 };
  try { return JSON.parse(localStorage.getItem(`rag_baseline_${ticker}`)||'{"mean":0,"std":1,"n":0}'); } catch(e){ return { mean:0, std:1, n:0 }; }
}
function saveScoreBaseline(ticker, obj) {
  try { localStorage.setItem(`rag_baseline_${ticker}`, JSON.stringify(obj)); } catch(e){}
}
function computeZ(score, baseline) {
  const mean = baseline?.mean||0; const std = baseline?.std||1; const s = std>0.0001? (score-mean)/std : 0.0; return s.toFixed(2);
}
function updateScoreBaseline(ticker, scores) {
  if (!ticker || !scores || !scores.length) return;
  const prev = loadScoreBaseline(ticker);
  const all = (prev.samples||[]).concat(scores).slice(-200);
  const mean = all.reduce((a,b)=>a+b,0)/all.length;
  const variance = all.reduce((a,b)=>a+(b-mean)*(b-mean),0)/all.length;
  const std = Math.sqrt(variance)||1;
  saveScoreBaseline(ticker, { mean, std, n: all.length, samples: all });
}

function showToast(docCount) {
  const el = document.getElementById('updatesToast');
  const dc = document.getElementById('docCounts');
  if (!el || !dc) return;
  dc.textContent = `${docCount} docs`;
  el.style.display = 'block';
  setTimeout(()=>{ el.style.display='none'; }, 2000);
}

const refreshAnswerBtn = document.getElementById('refreshAnswerBtn');
const toggleSources = document.getElementById('toggleSources');
const copyAnswerJsonBtn = document.getElementById('copyAnswerJson');
const downloadAnswerJsonBtn = document.getElementById('downloadAnswerJson');

if (askBtn) {
  askBtn.addEventListener('click', async () => {
    const q = (qaInput && qaInput.value || '').trim();
    if (!q) { appendBot('Enter a question to ask.'); return; }
    const sent = lastResult?.sentiment;
    const risk = lastResult?.risk;
    if (!sent || !risk) { appendBot('Run an analysis first to build context.'); return; }
    const ticker = lastResult?.financial?.ticker;
    if (ticker) {
      const rebuilt = await rebuildVectorIndex(ticker, sent, risk);
      if (rebuilt) showToast(rebuilt.count);
    }
    const ans = await ragAnswerFromContext({ question: q, sentiment: sent, risk, k:5 });
    renderAnswerPanel(q, ans.top||[], { usedEmb: ans.usedEmb });
    lastAnswerPayload = { question: q, retriever: ans.usedEmb ? 'embeddings' : 'tfidf', ranked: (ans.top||[]).map((t,i)=>({ idx:i+1, id:t.doc.id, score:t.score, meta:t.doc.meta, text:t.doc.text })), answer: ans.answer };
  });
}

if (refreshAnswerBtn) {
  refreshAnswerBtn.addEventListener('click', async () => {
    const q = (qaInput && qaInput.value || '').trim();
    const sent = lastResult?.sentiment; const risk = lastResult?.risk;
    if (!q || !sent || !risk) return;
    const ans = await ragAnswerFromContext({ question: q, sentiment: sent, risk, k:5 });
    renderAnswerPanel(q, ans.top||[], { usedEmb: ans.usedEmb });
    lastAnswerPayload = { question: q, retriever: ans.usedEmb ? 'embeddings' : 'tfidf', ranked: (ans.top||[]).map((t,i)=>({ idx:i+1, id:t.doc.id, score:t.score, meta:t.doc.meta, text:t.doc.text })), answer: ans.answer };
  });
}

if (toggleSources) {
  toggleSources.addEventListener('change', () => {
    const q = (qaInput && qaInput.value || '').trim();
    const sent = lastResult?.sentiment; const risk = lastResult?.risk;
    if (!q || !sent || !risk) return;
    ragAnswerFromContext({ question: q, sentiment: sent, risk, k:5 }).then(ans=>{
      renderAnswerPanel(q, ans.top||[], { usedEmb: ans.usedEmb });
      lastAnswerPayload = { question: q, retriever: ans.usedEmb ? 'embeddings' : 'tfidf', ranked: (ans.top||[]).map((t,i)=>({ idx:i+1, id:t.doc.id, score:t.score, meta:t.doc.meta, text:t.doc.text })), answer: ans.answer };
    });
  });
}

let lastAnswerPayload = null;
if (copyAnswerJsonBtn) {
  copyAnswerJsonBtn.addEventListener('click', async () => {
    if (!lastResult) { appendBot('No answer yet.'); return; }
    try { await navigator.clipboard.writeText(JSON.stringify(lastAnswerPayload||{}, null, 2)); appendBot('Answer JSON copied.'); } catch(e) { appendBot('Copy failed.'); }
  });
}
if (downloadAnswerJsonBtn) {
  downloadAnswerJsonBtn.addEventListener('click', () => {
    if (!lastResult) { appendBot('No answer yet.'); return; }
    const blob = new Blob([JSON.stringify(lastAnswerPayload||{}, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    const t = lastResult?.financial?.ticker || 'answer';
    a.href=url; a.download=`answer_${t}_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.json`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  });
}

// Copy/Download/Share/Clear actions
const copyBtn = document.getElementById('copyBtn');
const dlBtn = document.getElementById('downloadBtn');
const shareBtn = document.getElementById('shareBtn');
const clearBtn = document.getElementById('clearBtn');

copyBtn.addEventListener('click', async () => {
  if (!lastResult) { appendBot('No results to copy yet.'); return; }
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
    appendBot('Results JSON copied to clipboard.');
  } catch (e) {
    appendBot('Clipboard copy failed.');
  }
});

dlBtn.addEventListener('click', () => {
  if (!lastResult) { appendBot('No results to download yet.'); return; }
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const t = lastResult?.financial?.ticker || 'result';
  a.href = url; a.download = `analysis_${t}_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.json`;
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
});

shareBtn.addEventListener('click', async () => {
  const t = input.value.trim() || lastResult?.financial?.ticker || '';
  const v = validateTicker(t); if (!v.ok) { appendBot('Enter a valid ticker to share.'); return; }
  const url = location.origin + location.pathname + '#' + encodeURIComponent(v.value);
  try {
    await navigator.clipboard.writeText(url);
    appendBot('Sharable link copied to clipboard.');
  } catch (e) {
    appendBot(`Link: ${url}`);
  }
});

clearBtn.addEventListener('click', () => { chatbox.innerHTML=''; lastResult=null; });

// Settings and live polling
function loadSettings() { try { return JSON.parse(localStorage.getItem('rtfe_settings')||'{}'); } catch(e) { return {}; } }
function saveSettings(s) { try { localStorage.setItem('rtfe_settings', JSON.stringify(s)); } catch(e){} }
window.__settings = loadSettings();

const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const newsUrl = document.getElementById('newsUrl');
const socialUrl = document.getElementById('socialUrl');
const pollInterval = document.getElementById('pollInterval');
const saveSettingsBtn = document.getElementById('saveSettings');
const closeSettingsBtn = document.getElementById('closeSettings');

function openSettings() {
  newsUrl.value = window.__settings.newsUrl || '';
  socialUrl.value = window.__settings.socialUrl || '';
  pollInterval.value = window.__settings.pollInterval || 30;
  settingsPanel.style.display = 'block';
}
function closeSettings() { settingsPanel.style.display = 'none'; }
settingsBtn.addEventListener('click', openSettings);
closeSettingsBtn.addEventListener('click', closeSettings);
saveSettingsBtn.addEventListener('click', () => {
  window.__settings = { newsUrl: newsUrl.value.trim(), socialUrl: socialUrl.value.trim(), pollInterval: Math.max(5, parseInt(pollInterval.value, 10) || 30) };
  saveSettings(window.__settings);
  appendBot('Settings saved.');
  closeSettings();
});

const liveBtn = document.getElementById('liveBtn');
let liveOn = false;
function setLiveLabel() { liveBtn.textContent = `Live: ${liveOn ? 'On' : 'Off'}`; }
async function startLive() {
  const v = validateTicker(input.value.trim() || lastResult?.financial?.ticker || '');
  if (!v.ok) { appendBot('Enter a valid ticker to start live mode.'); liveOn=false; setLiveLabel(); return; }
  const ticker = v.value;
  clearLive();
  await runAnalysisOnce(ticker);
  const intervalMs = (window.__settings?.pollInterval || 30) * 1000;
  liveTimer = setInterval(() => { runAnalysisOnce(ticker); }, intervalMs);
}
liveBtn.addEventListener('click', async () => {
  liveOn = !liveOn; setLiveLabel();
  if (liveOn) { await startLive(); } else { clearLive(); }
});

// Auto-run if hash contains a ticker
window.addEventListener('load', () => {
  const t = getHashTicker();
  if (t) { input.value = t; analyze(); }
  setLiveLabel();
});


