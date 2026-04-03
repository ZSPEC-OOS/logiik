# Logiik

Scientific reasoning AI framework. Trains a model through
12 progressive curriculum phases from memorization to
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

| Phase | Track | Name | Gen. Ratio | T-S |
|-------|-------|------|-----------|-----|
| 1 | Foundation | Memorization | 10% | |
| 2 | Foundation | Generation | 50% | |
| 3 | Language | Scientific Language & Literature | 65% | |
| 4 | Language | Mathematical & Statistical Reasoning | 70% | |
| 5 | Domain | Scientific Reasoning & Experimental Design | 88% | |
| 6 | Domain | Niche & Interdisciplinary Scientific Reasoning | 93% | ✓ |
| 7 | Domain | Scientific Image & Data Analysis | 93% | |
| 8 | Execution | Research Computing & Scientific Coding | 92% | |
| 9 | Execution | Engineering Execution & Reliability | 90% | |
| 10 | Integration | Abstraction & Cross-Domain Synthesis | 94% | |
| 11 | Integration | Adversarial Robustness & Epistemic Integrity | 95% | ✓ |
| 12 | Capstone | Synthetic Judgment (PPO/TRL) | 95% | |

T-S = Teacher-Student iterative feedback loop active.

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
