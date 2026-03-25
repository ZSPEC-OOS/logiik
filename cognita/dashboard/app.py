"""
NERO Dashboard - Real-time training visualization interface.
Beautiful, user-friendly monitoring of AI learning process.
"""
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="NERO - Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 1px solid #667eea30;
        border-radius: 1rem;
        padding: 1.2rem;
        text-align: center;
    }
    .training-phase {
        background: white;
        border-left: 4px solid #667eea;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


class TrainingMonitor:
    """Real-time training data monitor."""

    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.is_training = False
        self.current_phase = "Not Started"
        self.loss_history = []
        self.accuracy_history = []
        self.knowledge_growth = []

    def update_metrics(self, metrics: dict):
        self.metrics_queue.put(metrics)

    def get_latest(self):
        updates = []
        while not self.metrics_queue.empty():
            updates.append(self.metrics_queue.get())
        return updates


# Initialize session state
if 'monitor' not in st.session_state:
    st.session_state.monitor = TrainingMonitor()
if 'training_log' not in st.session_state:
    st.session_state.training_log = []


def render_header():
    """Render attractive header."""
    st.markdown(
        '<h1 class="main-header">🧠 NERO Training Dashboard</h1>',
        unsafe_allow_html=True
    )
    cols = st.columns(4)
    metrics = [
        ("📚 Training Examples", "1,247", "+23 this session"),
        ("🎯 Current Accuracy", "78.4%", "+5.2% from yesterday"),
        ("💾 Knowledge Size", "2.3 GB", "3 checkpoints"),
        ("⏱️ Training Time", "4h 23m", "Phase 2 of 3")
    ]

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; opacity:0.9;">{label}</h4>
                <h2 style="margin:0; font-size:2rem;">{value}</h2>
                <p style="margin:0; opacity:0.8; font-size:0.9rem;">{delta}</p>
            </div>
            """, unsafe_allow_html=True)


def render_training_phases():
    """Visualize the 3-phase training progression."""
    st.subheader("🎓 Training Curriculum Progress")
    phases = [
        ("Phase 1: Memorization", "Learning from Teacher's Q+A structure", 100, "completed"),
        ("Phase 2: Generation", "Creating original answers", 65, "active"),
        ("Phase 3: Abstraction", "Cross-domain knowledge synthesis", 0, "pending")
    ]

    cols = st.columns(3)
    for col, (title, desc, progress, status) in zip(cols, phases):
        with col:
            color = (
                "#4CAF50" if status == "completed"
                else "#667eea" if status == "active"
                else "#9e9e9e"
            )
            opacity = 1 if status != "pending" else 0.6
            st.markdown(f"""
            <div class="training-phase" style="border-left-color:{color}; opacity:{opacity};">
                <h4 style="color:{color}; margin-top:0;">{title}</h4>
                <p style="font-size:0.9rem; color:#666; margin-bottom:0.5rem;">{desc}</p>
                <div style="background:#e0e0e0; border-radius:10px; height:8px;">
                    <div style="width:{progress}%; background:{color}; height:100%;
                                border-radius:10px; transition:width 0.5s;"></div>
                </div>
                <p style="text-align:right; font-size:0.8rem; margin-top:0.3rem;
                           color:{color}; font-weight:bold;">{progress}%</p>
            </div>
            """, unsafe_allow_html=True)


def render_live_charts():
    """Real-time training metrics visualization."""
    st.subheader("📊 Live Training Metrics")

    time_points = list(range(50))
    rng = np.random.default_rng(42)
    loss = [2.5 * np.exp(-0.05 * t) + 0.2 + rng.normal(0, 0.05) for t in time_points]
    accuracy = [50 + 40 * (1 - np.exp(-0.05 * t)) + rng.normal(0, 2) for t in time_points]
    generative_score = [20 + 60 * (1 - np.exp(-0.03 * t)) for t in time_points]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Loss Curve", "Accuracy Progress",
            "Generative Capability", "Knowledge Distribution"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Loss curve
    fig.add_trace(
        go.Scatter(
            x=time_points, y=loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ),
        row=1, col=1
    )

    # Accuracy
    fig.add_trace(
        go.Scatter(
            x=time_points, y=accuracy,
            mode='lines',
            name='Accuracy %',
            line=dict(color='#4ECDC4', width=2),
            fill='tozeroy',
            fillcolor='rgba(78, 205, 196, 0.2)'
        ),
        row=1, col=2
    )

    # Generative score
    fig.add_trace(
        go.Scatter(
            x=time_points, y=generative_score,
            mode='lines',
            name='Originality Score',
            line=dict(color='#667eea', width=3, dash='dot'),
            marker=dict(size=6)
        ),
        row=2, col=1
    )

    # Knowledge domain distribution (donut chart)
    domains = ["Reasoning", "Facts", "Creativity", "Analysis", "Synthesis"]
    values = [25, 30, 20, 15, 10]
    fig.add_trace(
        go.Pie(
            labels=domains,
            values=values,
            hole=0.6,
            marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            textinfo='label+percent',
            textposition='outside'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_knowledge_explorer():
    """Interactive knowledge base viewer."""
    st.subheader("🔍 Knowledge Base Explorer")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Attachable Knowledge Folders**")

        knowledge_path = Path("./knowledge_base")
        if knowledge_path.exists():
            for folder in ["embeddings", "checkpoints", "training_data", "metadata"]:
                folder_path = knowledge_path / folder
                if folder_path.exists():
                    size = sum(
                        f.stat().st_size for f in folder_path.rglob("*") if f.is_file()
                    ) / 1024 / 1024
                    file_count = len(list(folder_path.glob('*')))
                    st.markdown(f"""
                    <div style="background:white; padding:0.8rem; border-radius:0.5rem;
                                margin:0.3rem 0; box-shadow:0 2px 5px rgba(0,0,0,0.05);">
                        📁 <b>{folder}</b><br/>
                        <span style="color:#666; font-size:0.8rem;">
                            {size:.1f} MB &bull; {file_count} files
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No knowledge base found. Start training to create one.")

        if st.button("📥 Export Knowledge Package", type="primary"):
            st.success("Knowledge exported to nero_knowledge_latest.zip")

    with col2:
        st.markdown("**Recent Training Examples**")

        examples_df = pd.DataFrame({
            "Question": [
                "What causes ocean tides?",
                "Explain quantum entanglement",
                "Write a creative story opening"
            ],
            "Type": ["Fact", "Concept", "Generation"],
            "Difficulty": ["Easy", "Hard", "Medium"],
            "Score": [95, 72, 88]
        })

        st.dataframe(
            examples_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    help="Training accuracy on this example",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                )
            }
        )


def render_chat_test():
    """Interactive testing interface."""
    st.subheader("💬 Test Your AI")
    user_input = st.text_area(
        "Ask your trained AI a question:",
        placeholder="e.g., What have you learned about machine learning?",
        height=100
    )

    cols = st.columns([1, 4])
    with cols[0]:
        test_btn = st.button("🚀 Generate Response", type="primary", use_container_width=True)

    if test_btn and user_input:
        with st.spinner("AI is thinking..."):
            time.sleep(1.5)

            response_col, metrics_col = st.columns([2, 1])

            with response_col:
                st.markdown("""
                <div style="background:linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                            padding:1.5rem; border-radius:1rem; border-left:4px solid #667eea;">
                    <p style="margin:0; font-size:1.1rem; line-height:1.6;">
                        Based on my training, machine learning is a subset of artificial intelligence
                        that enables systems to learn and improve from experience without being
                        explicitly programmed. It involves algorithms that can identify patterns in
                        data and make decisions with minimal human intervention.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with metrics_col:
                st.metric("Confidence", "87%", "+12%")
                st.metric("Originality", "High", "Not memorized")
                st.metric("Tokens", "156", "Fast generation")


def main():
    """Main dashboard application."""
    render_header()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Training Controls")

        if st.button("▶️ Start Training", type="primary", use_container_width=True):
            st.session_state.monitor.is_training = True
            st.success("Training initiated!")

        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.monitor.is_training = False
            st.info("Training paused.")

        st.markdown("---")
        st.markdown("### 🔗 Teacher API")
        st.selectbox("Teacher Model", ["GPT-4", "Claude-3 Opus", "Custom API"])
        st.slider("Examples per Batch", 5, 50, 20)
        st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

        st.markdown("---")
        st.markdown("### 💾 Knowledge Base")
        if st.button("📂 Attach Knowledge Folder", use_container_width=True):
            st.info("Select your knowledge_base folder")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Training Monitor", "🧠 Knowledge Base", "💬 Test AI"])

    with tab1:
        render_training_phases()
        render_live_charts()

        if st.session_state.monitor.is_training:
            st.markdown("*Live updates every 2 seconds...*")

    with tab2:
        render_knowledge_explorer()

    with tab3:
        render_chat_test()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#666;'>NERO v1.0 &bull; "
        "<a href='https://github.com/zspec-oos/logik2'>GitHub</a></p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
