# 🧠 NERO

A complete AI training framework implementing modern teacher-student architecture with local, attachable knowledge storage.

## 🎯 Key Features

- **Teacher-Student Architecture**: Connect to GPT-4/Claude APIs as teachers
- **Structured Training**: Question + 5-10 answers curriculum format
- **Generative Capabilities**: AI generates original answers, not just memorization
- **Local Knowledge Base**: All learning stored in attachable `knowledge_base/` folder
- **Real-time Dashboard**: Beautiful Streamlit interface for monitoring training
- **REST API + WebSocket**: FastAPI server for training control and live updates
- **Frontier Migration Kit (March 2026)**: MoE-ready self-hosted runtime + LangGraph orchestration for coding and biological aging discovery

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env and set your TEACHER_API_KEY

# Start API server
python -m cognita.api.server

# Launch dashboard (new terminal)
streamlit run cognita/dashboard/app.py
```

## ⚙️ Frontier Migration Protocol (NERO)

Use this path to upgrade NERO from legacy dense local models to a frontier-aligned open-weight MoE architecture.

### 1) Run migration bootstrap

```bash
bash scripts/migrate_frontier_stack.sh
# Optional: fetch large model weights too
bash scripts/migrate_frontier_stack.sh --with-ollama-pull
```

This script backs up the current repo, installs vLLM/LangGraph/LlamaIndex dependencies, installs CUDA 12.4 PyTorch wheels, and prepares Ollama model runtime.

### 2) Review migration config

- Main profile: `configs/frontier_stack.yaml`
- Set backend (`ollama` or `vllm`), context goals (128K–10M), agent roles, RAG corpus path, and LoRA specialization settings.

### 3) Start a frontier inference backend

**Ollama path (simplest):**
```bash
ollama serve
```

**vLLM path (higher throughput):**
```bash
docker compose up vllm
# OpenAI-compatible endpoint on http://localhost:8001/v1
```

### 4) Run stateful agentic workflow

```bash
python -m cognita.orchestration.bio_coding_graph
```

This runs a dual-agent (researcher + coder) LangGraph cycle for biological-aging hypothesis synthesis and code planning.

### 5) Add RAG corpus

Place papers/notes under:

```text
knowledge_base/aging_papers/
```

Then wire your query engine in orchestration nodes (placeholder hooks are included).

## 🐳 Docker

```bash
# Set your API key
export TEACHER_API_KEY="your-api-key"

# Start all services
docker-compose up

# API available at: http://localhost:8000
# Dashboard at:     http://localhost:8501
```

## 📁 Repository Structure

```
cognita/
├── core/
│   ├── brain.py              # Transformer + LoRA + Generative head
│   └── teacher_interface.py  # OpenAI & Anthropic teacher connectors
├── training/
│   └── curriculum.py         # 5-phase curriculum engine
├── storage/
│   └── checkpoint_manager.py # Local knowledge base manager
├── dashboard/
│   └── app.py                # Streamlit real-time dashboard
├── orchestration/
│   └── bio_coding_graph.py   # LangGraph workflow for coding + aging
└── api/
    └── server.py             # FastAPI + WebSocket server

configs/
├── model_config.yaml         # Model & training hyperparameters
├── teacher_config.yaml       # Teacher API configuration
└── frontier_stack.yaml       # Frontier migration target profile

scripts/
└── migrate_frontier_stack.sh # One-shot migration bootstrap

knowledge_base/               # Attachable AI knowledge folder
├── embeddings/               # Vector representations for RAG
├── checkpoints/              # Model snapshots
├── training_data/            # Training session history
└── metadata/                 # Indices & configuration
```

## 🎓 Training Phases

| Phase | Name | Description | Generative Ratio |
|-------|------|-------------|-----------------|
| 1 | **Memorization** | Learn from teacher's Q+A structure | 10% |
| 2 | **Generation** | Create original answers | 50% |
| 3 | **Abstraction** | Cross-domain knowledge synthesis | 80% |
| 4 | **Coding Mastery** | Complete coding understanding across common languages | 90% |
| 5 | **Drosophila AI Framework** | Specialized framework design for Drosophila genetics with axon guidance and neural wiring focus | 95% |

See `docs/PHASE_CURRICULUM_REVIEW.md` for a detailed readiness review and a recommended 7-phase progression that adds two bridge phases before advanced coding/scientific specialization.

## 🏗️ Architecture

```
Teacher API (GPT-4 / Claude)
        │
        ▼ Q + 5-10 Answers
┌───────────────────────────────┐
│           NEROBrain            │
│  ┌─────────────────────────┐  │
│  │  Base Transformer (LM)  │  │
│  │  + LoRA Adapters        │  │
│  └───────────┬─────────────┘  │
│              │ hidden states   │
│  ┌───────────▼─────────────┐  │
│  │    Generative Head      │  │
│  │ (original answer synth) │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
        │
        ▼
knowledge_base/ (attachable)
```

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/initialize` | POST | Initialize brain + teacher |
| `/train/start` | POST | Begin training |
| `/train/stop` | POST | Stop training |
| `/ask` | POST | Query the trained AI |
| `/knowledge/summary` | GET | Knowledge base stats |
| `/knowledge/export` | POST | Export knowledge package |
| `/ws/training` | WebSocket | Live training metrics |

## 🔧 Configuration

Edit `configs/model_config.yaml` to customize:
- Base model (default: `microsoft/DialoGPT-medium`)
- LoRA rank and alpha
- Training batch size and learning rate
- Curriculum phase durations

Edit `configs/teacher_config.yaml` to configure:
- Teacher provider (`openai` or `anthropic`)
- Examples per topic and difficulty range
- Evaluation criteria

## 📦 Knowledge Portability

The `knowledge_base/` folder is fully portable. To transfer a trained model:

```python
from cognita.storage.checkpoint_manager import KnowledgeBaseManager

manager = KnowledgeBaseManager("./knowledge_base")

# Export
manager.export_knowledge_package("./exports", "my_model_v1")

# Import on another machine
manager.import_knowledge_package("./exports/nero_knowledge_my_model_v1.zip")
```
