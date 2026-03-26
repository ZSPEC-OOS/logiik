# Migration Protocol: NERO → Frontier-Aligned Open-Weight MoE Stack

Version 1.0 (adapted for NERO) | March 2026

This protocol upgrades NERO from a legacy dense-model workflow to a local/hybrid frontier architecture with:

- Decoder-only MoE backbones.
- Extended-context operation (128K baseline, up to 10M where backend/model supports it).
- Tool-calling and stateful multi-agent orchestration.
- Coding + biological-aging discovery workflow support.

## 1) Scope and prerequisites

### Objectives

- Improve code-generation + reasoning performance with frontier open-weight MoE models.
- Enable autonomous, stateful multi-agent runs for literature synthesis, hypothesis design, and code validation.
- Preserve NERO’s local knowledge-base portability.

### Hardware

- Minimum: 2× RTX 4090/A6000 (quantized inference profile).
- Recommended: 4–8× H100/A100 for high-throughput MoE inference.
- Storage: 2 TB+ NVMe for model caches, vector indexes, and checkpoints.

### Software

- Ubuntu 22.04/24.04 (recommended) or macOS 15+.
- CUDA 12.4+, cuDNN 9.x, Python 3.11+.

## 2) Infrastructure migration

Use the project helper:

```bash
bash scripts/migrate_frontier_stack.sh
```

What it does:

1. Backs up the current NERO repo.
2. Installs vLLM + LangGraph + LlamaIndex + related packages.
3. Installs CUDA 12.4 PyTorch wheels.
4. Installs Ollama if absent.
5. Optionally pulls frontier models (`--with-ollama-pull`).

## 3) Model acquisition and deployment

Primary profile in `configs/frontier_stack.yaml`.

### Ollama path

```bash
ollama serve
ollama pull llama4:maverick
ollama pull deepseek-v3.2
```

### vLLM path

Use provided Docker service:

```bash
docker compose up vllm
```

Endpoint: `http://localhost:8001/v1`.

## 4) Agentic orchestration (LangGraph)

Run the included template:

```bash
python -m cognita.orchestration.bio_coding_graph
```

Current template includes:

- Researcher node (hypothesis + design synthesis).
- Coder node (implementation plan).
- Tool placeholders for code execution and PubMed retrieval.
- Thread-aware invocation plus local SQLite checkpoint target.

## 5) RAG and private-domain integration

- Put domain corpus in `knowledge_base/aging_papers`.
- Connect LlamaIndex/Chroma retrieval inside LangGraph nodes.
- Use top-k and context limits in `configs/frontier_stack.yaml`.

## 6) Fine-tuning/specialization

Use LoRA/PEFT settings from `configs/frontier_stack.yaml`:

- `lora_r: 16`
- `lora_alpha: 32`
- `sequence_len: 131072`

Training data paths are staged under `knowledge_base/training_data`.

## 7) Validation and productionization

Track targets (configured):

- SWE-Bench Verified `>=70`
- LiveCodeBench `>=80`
- GPQA Diamond (tracked scientific reasoning)

Deploy options:

- Single host: Ollama or vLLM + NERO API.
- Multi-node: Kubernetes + inference service + persistent vector/checkpoint volumes.

## Troubleshooting

- **OOM during MoE inference:** lower context, increase tensor parallelism, or enable stronger quantization.
- **Tool-call hallucination:** enforce Pydantic/structured schema on tool responses.
- **Context failures:** lower retrieval fanout or tune max model length.

## Notes

- This migration package is intentionally scaffolded for safe adoption in NERO and should be iterated per environment.
- Replace placeholder PubMed tooling and constrained code execution with production-grade sandboxed adapters before unattended execution.
