"""LangGraph orchestration for NERO coding + biological aging workflows.

This module provides a runnable template for stateful, multi-agent execution
using an OpenAI-compatible endpoint (Ollama or vLLM).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


class WorkflowState(TypedDict, total=False):
    query: str
    messages: List[str]


def _default_sqlite_path() -> str:
    return str(Path(tempfile.gettempdir()) / "nero_langgraph_checkpoint.sqlite")


def _safe_python_exec(code: str) -> str:
    """Very restricted code execution helper for controlled experiments.

    NOTE: This remains intentionally constrained and should be moved to a
    containerized sandbox for production use.
    """
    safe_builtins = {
        "len": len,
        "range": range,
        "sum": sum,
        "min": min,
        "max": max,
    }
    local_vars: Dict[str, Any] = {}
    exec(code, {"__builtins__": safe_builtins}, local_vars)  # noqa: S102
    return str(local_vars.get("result", "Executed"))


@tool("code_exec")
def code_exec(code: str) -> str:
    """Execute constrained Python snippets for bioinformatics experimentation."""
    return _safe_python_exec(code)


@tool("pubmed_search")
def pubmed_search(query: str) -> str:
    """PubMed placeholder for migration phase; replace with API/local index adapter."""
    return f"Placeholder result for query: {query}"


def build_graph(model: str = "llama4:maverick", temperature: float = 0.2):
    """Build a dual-agent LangGraph app with persistence-ready state transitions."""
    llm = ChatOllama(model=model, temperature=temperature)

    def researcher_node(state: WorkflowState) -> WorkflowState:
        query = state.get("query", "")
        msg = llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Analyze this aging hypothesis and propose testable code: "
                        f"{query}. Include expected biomarkers and evaluation plan."
                    )
                )
            ]
        )
        return {"messages": state.get("messages", []) + [msg.content]}

    def coder_node(state: WorkflowState) -> WorkflowState:
        prompt = state.get("messages", ["No prior context"])[-1]
        msg = llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Implement a concise Python prototype and validation checklist "
                        "for this plan:\n"
                        f"{prompt}"
                    )
                )
            ]
        )
        return {"messages": state.get("messages", []) + [msg.content]}

    graph = StateGraph(WorkflowState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", END)

    return graph.compile()


def run_once(query: str, thread_id: str = "aging_exp_001") -> Dict[str, Any]:
    """Execute one full orchestration cycle."""
    app = build_graph(
        model=os.getenv("NERO_FRONTIER_MODEL", "llama4:maverick"),
        temperature=float(os.getenv("NERO_FRONTIER_TEMPERATURE", "0.2")),
    )

    # Create a local sqlite file so the migration has a concrete persistence target.
    sqlite_path = os.getenv("NERO_LANGGRAPH_SQLITE", _default_sqlite_path())
    sqlite3.connect(sqlite_path).close()

    result = app.invoke(
        {"query": query, "messages": []},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {"thread_id": thread_id, "sqlite_path": sqlite_path, "result": result}


if __name__ == "__main__":
    payload = run_once("Telomere extension via partial epigenetic reprogramming")
    print(payload)
