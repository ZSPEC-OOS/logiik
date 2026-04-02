# Logiik

Scientific reasoning AI framework. Trains a model through
10 progressive curriculum phases from memorization to
synthetic judgment under uncertainty.

---

## Quick Start

### 1. Install dependencies
pip install -r requirements.txt

### 2. Configure credentials
cp .env.example .env
# Edit .env — fill in PINECONE_API_KEY and FIREBASE_API_KEY

### 3. Verify environment
python logiik/main.py --mode check

### 4. Run tests
python logiik/main.py --mode test

### 5. Start API + dashboard
python logiik/main.py --mode api
# Open http://localhost:8001 in browser

### 6. Start GPU session (when GPU instance is running)
python logiik/main.py --mode session
# Send queries: POST http://localhost:5000/ask

---

## Curriculum Phases

| Phase | Name | Generative Ratio |
|-------|------|-----------------|
| 1 | Memorization | 10% |
| 2 | Generation | 50% |
| 3 | Abstraction | 80% |
| 4 | Engineering Execution & Reliability | 85% |
| 5 | Coding Mastery | 90% |
| 6 | Scientific Reasoning & Experimental Design | 93% |
| 7 | Niche Scientific Reasoning | 94% |
| 8 | Scientific Image Analysis | 94% |
| 9 | PDF / Textbook Ingestion | 95% |
| 10 | Synthetic Judgment | 95% |

---

## Architecture

```
logiik/
  core/           Training loops, generation, Phase 10 PPO
  storage/        Pinecone, Firebase, Redis
  embeddings/     SPECTER2 (text), BLIP-2 (images)
  retrieval/      RAG pipeline
  ingestion/      Phase 8 (images), Phase 9 (PDFs)
  curriculum/     Phase definitions and config
  utils/          Logging, helpers, env check
  api/            FastAPI endpoints
  dashboard/      Web dashboard
  session_manager/ GPU session manager
  tests/          Full test suite
```

---

## Services

| Service | Status | Purpose |
|---------|--------|---------|
| Pinecone | Required | Vector embeddings (dim=768) |
| Firebase | Required | Full-text storage (REST API) |
| Redis | Optional | Hot cache (disabled by default) |
| AWS S3 | Deferred | Model weights storage |
| GPU | Deferred | Training and inference |

---

## Pre-Deployment Security Checklist

Before storing real training data or model weights:

- [ ] Rotate Pinecone API key
- [ ] Rotate Firebase API key
- [ ] Confirm .env is in .gitignore
- [ ] Set Firebase security rules
- [ ] Set Pinecone index access controls
- [ ] Configure AWS S3 bucket (when ready)
- [ ] Set GPU instance firewall rules

---

## Legacy

This project was previously named NERO / Cognita.
Legacy code is preserved at _legacy_backup/.
