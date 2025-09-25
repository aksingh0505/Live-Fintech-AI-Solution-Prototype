Real-Time Financial Expert (Static SPA)

A deployable, client-only web app that simulates financial health, media sentiment, and risk analysis for a stock ticker. Works entirely in the browser with mock models, so it can be hosted on GitHub Pages without a backend.

Live Hosting (GitHub Pages)
1. Push this repository to GitHub.
2. In your repository: Settings → Pages → Build and deployment → Source: Deploy from a branch.
3. Select the main branch and root folder (/). Save.
4. Open the URL shown in the Pages section. The app is served from index.html at the repo root.

Usage
- Enter a ticker like AAPL and click Analyze.
- Share results via deep link (URL hash). Example: #/AAPL.
- Export results:
  - Copy JSON: copies to clipboard
  - Download JSON: saves a analysis_TICKER_timestamp.json

Ask the Ticker (Dynamic RAG)
- Right-side pane for grounded Q&A over live headlines/social.
- Retrieval: Pathway-powered backend when configured, with in-browser embeddings as fallback and TF‑IDF as last resort.
- Chunking: 256–384 token windows with overlap; per-chunk metadata (title/source/time/type).
- Vector index: lightweight FAISS-like index in browser (IndexedDB) with in-memory LRU of top 1,000 chunks per ticker.
- Live mode: while Live is On, polling refreshes context; an "index updated" toast appears with doc counts. Answers re-rank only on Refresh.
- Guardrails: per-ticker z-scores vs session baseline; risk topic tags shown as chips.
- Export: Copy/Download JSON near the answer exports ranked set, IDs, and final answer.

Powered by Pathway

Backend (optional, for enhanced retrieval):

1. Install dependencies:
   pip install -U pathway fastapi uvicorn sentence-transformers

2. Run the service:
   uvicorn backend.main:app --host 0.0.0.0 --port 8000

   - Docker: build and run
     docker build -t rtfe-pathway backend
     docker run --rm -p 8000:8000 rtfe-pathway

3. In the app Settings, set Pathway API URL to your server (e.g., http://localhost:8000).

Endpoints:
- POST /ingest { ticker, docs: [{ id, title, text, source, published_at }] } -> { upserts, duplicates, total }
- POST /query { ticker, question, k } -> { chunks: [{ id, title, text, source, published_at, score }], rationale: "pathway", used: "pathway" }
- GET /health -> { ok: true }

SPA Integration

- The Ask panel first attempts the Pathway /query with a ~2s timeout. On success, the status pill shows "Pathway: On" and results are rendered. On error/timeout/empty results, it falls back to local embeddings, and then TF‑IDF if embeddings are unavailable.
- During Analyze/Live runs, headlines and social posts are upserted to the Pathway backend in the background (errors ignored). This keeps the backend index fresh.
- JSON exports include retriever: "pathway" | "embeddings" | "tfidf".

Disclaimers
This is a heuristic mock. Not investment advice.

Local Development
Open index.html in a browser. No build step required.

Python Utilities (Optional)
Two enhanced CLI tools simulate server-side logic locally:

- Financial model
  python market.py --ticker AAPL --json
- Sentiment and risk
  python sentiment.py --ticker AAPL --risk --json

These do not power the web app (which is client-only) but are useful for testing.

Customization
- Edit scoring heuristics and mock data in index.html JavaScript.
- Replace mocks with real APIs by adding fetch calls and server-side proxies (not GitHub Pages compatible).

Security & Deploy Notes
- Keep API keys off the client. If using real news/social APIs, add a tiny proxy with rate limiting and CORS; otherwise stick to mock endpoints as configured in Settings.
- Feature‑detect embeddings/LLM; the app degrades gracefully to TF‑IDF and heuristic synthesis.

License
MIT

