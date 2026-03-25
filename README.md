# 🧠 NERO

A complete AI training framework implementing modern teacher-student architecture with local, attachable knowledge storage.

## 🎯 Key Features

- **Teacher-Student Architecture**: Connect to GPT-4/Claude APIs as teachers
- **Structured Training**: Question + 5-10 answers curriculum format
- **Generative Capabilities**: AI generates original answers, not just memorization
- **Local Knowledge Base**: All learning stored in attachable `knowledge_base/` folder
- **Real-time Dashboard**: Beautiful Streamlit interface for monitoring training
- **REST API + WebSocket**: FastAPI server for training control and live updates

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
│   └── curriculum.py         # 3-phase curriculum engine
├── storage/
│   └── checkpoint_manager.py # Local knowledge base manager
├── dashboard/
│   └── app.py                # Streamlit real-time dashboard
└── api/
    └── server.py             # FastAPI + WebSocket server

configs/
├── model_config.yaml         # Model & training hyperparameters
└── teacher_config.yaml       # Teacher API configuration

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
