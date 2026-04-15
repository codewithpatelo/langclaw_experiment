"""Streamlit real-time dashboard for the LangClaw simulation.

Run with:
    streamlit run dashboard.py

Panels:
    1. Interactive argument graph (pyvis).
    2. Agent deficit evolution (plotly line chart).
    3. Scrollable action log.
    4. Live metrics (τ-bench, nodes, edges, density).
    5. Utility scores per agent per tick (HRRL mode only).

Supports orchestration mode selection (hrrl / round-robin / random).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network

from langclaw.schemas import SimulationLog
from langclaw.simulation import OrchestrationMode, SotopiaEnvironment

load_dotenv()

AGENT_COLORS: dict[str, str] = {
    "GOV-1": "#3498db",
    "GOV-2": "#2980b9",
    "OPP-1": "#e74c3c",
    "OPP-2": "#c0392b",
}

AGENT_GROUPS: dict[str, str] = {
    "GOV-1": "Oficialismo",
    "GOV-2": "Oficialismo",
    "OPP-1": "Oposición",
    "OPP-2": "Oposición",
}


def init_session_state() -> None:
    if "env" not in st.session_state:
        st.session_state.env = None
        st.session_state.current_tick = 0
        st.session_state.all_logs: list[SimulationLog] = []
        st.session_state.deficit_history: dict[str, list[float]] = {
            r["id"]: [0.1] for r in _get_roles()
        }
        st.session_state.running = False
        st.session_state.initialised = False


def _get_roles() -> list[dict[str, str]]:
    from langclaw.simulation import AGENT_ROLES
    return AGENT_ROLES


def create_environment(
    base_url: str,
    model: str,
    api_key: str,
    iterations: int,
    orchestration_mode: str,
    api_hard_limit: int,
    initial_deficit: float,
) -> None:
    st.session_state.env = SotopiaEnvironment(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_iterations=iterations,
        orchestration_mode=OrchestrationMode(orchestration_mode),
        api_hard_limit=api_hard_limit,
        initial_deficit=initial_deficit,
    )
    st.session_state.current_tick = 0
    st.session_state.all_logs = []
    st.session_state.deficit_history = {
        agent.agent_id: [agent.drive.deficit] for agent in st.session_state.env.agents
    }
    st.session_state.initialised = True


def run_tick() -> list[SimulationLog]:
    env: SotopiaEnvironment = st.session_state.env
    st.session_state.current_tick += 1
    tick = st.session_state.current_tick

    tick_logs = env.run_single_tick(tick)
    st.session_state.all_logs.extend(tick_logs)

    for agent in env.agents:
        st.session_state.deficit_history[agent.agent_id].append(agent.drive.deficit)

    return tick_logs


def render_graph_panel(env: SotopiaEnvironment) -> None:
    st.subheader("Grafo de Argumentos")

    nodes = env.graph.get_all_nodes()
    edges = env.graph.get_all_edges()

    if not nodes:
        st.info("Aún no hay argumentos en el grafo.")
        return

    net = Network(
        height="420px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for node in nodes:
        label = node.get("claim", "")[:40]
        agent_id = node.get("agent_id", "")
        color = AGENT_COLORS.get(agent_id, "#888888")
        title = (
            f"[{node['id']}]\nAgent: {agent_id}\n"
            f"Tick: {node.get('tick', '?')}\n\n{node.get('claim', '')}"
        )
        net.add_node(
            node["id"],
            label=label,
            color=color,
            title=title,
            size=20,
            font={"size": 10, "color": "white"},
        )

    for edge in edges:
        attack = edge.get("attack_type", "rebuttal")
        color = "#f39c12" if attack == "undercut" else "#e74c3c"
        net.add_edge(
            edge["source"], edge["target"],
            color=color, title=attack, arrows="to", width=2,
        )

    html_path = Path("_graph.html")
    net.save_graph(str(html_path))
    st.components.v1.html(html_path.read_text(encoding="utf-8"), height=450, scrolling=True)


def render_deficit_panel() -> None:
    st.subheader("Evolución del Déficit Epistémico")

    history = st.session_state.deficit_history
    fig = go.Figure()

    for agent_id, values in history.items():
        fig.add_trace(go.Scatter(
            y=values,
            mode="lines+markers",
            name=f"{agent_id} ({AGENT_GROUPS.get(agent_id, '')})",
            line=dict(color=AGENT_COLORS.get(agent_id, "#888"), width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        xaxis_title="Tick",
        yaxis_title="Déficit",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.add_hline(
        y=0.7, line_dash="dash", line_color="rgba(255,255,255,0.3)",
        annotation_text="θ (umbral)", annotation_position="right",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_utility_panel() -> None:
    """Show the utility score breakdown for the most recent tick (HRRL only)."""
    logs: list[SimulationLog] = st.session_state.all_logs
    if not logs:
        return

    last_tick = logs[-1].tick
    tick_logs = [l for l in logs if l.tick == last_tick]

    if not any(l.utility_debate or l.utility_search or l.utility_read for l in tick_logs):
        return  # baseline mode — no utility scores

    st.subheader(f"Utilidad por Acción — Tick {last_tick}")

    agents = [l.agent_id for l in tick_logs]
    debate_vals = [l.utility_debate for l in tick_logs]
    search_vals = [l.utility_search for l in tick_logs]
    read_vals = [l.utility_read for l in tick_logs]
    pass_vals = [l.utility_pass for l in tick_logs]

    fig = go.Figure(data=[
        go.Bar(name="DEBATE", x=agents, y=debate_vals, marker_color="#e74c3c"),
        go.Bar(name="SEARCH", x=agents, y=search_vals, marker_color="#f39c12"),
        go.Bar(name="READ",   x=agents, y=read_vals,   marker_color="#2ecc71"),
        go.Bar(name="PASS",   x=agents, y=pass_vals,   marker_color="#95a5a6"),
    ])
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=280,
        margin=dict(l=30, r=20, t=20, b=40),
        legend=dict(orientation="h"),
        yaxis_title="Utilidad",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_log_panel() -> None:
    st.subheader("Log de Acciones")

    logs: list[SimulationLog] = st.session_state.all_logs
    if not logs:
        st.info("No hay acciones registradas aún.")
        return

    rows: list[dict[str, Any]] = []
    for entry in reversed(logs[-40:]):
        rows.append({
            "Tick": entry.tick,
            "Agente": entry.agent_id,
            "Acción": entry.action,
            "Déficit": f"{entry.deficit_after:.3f}",
            "Δφ*": f"{entry.delta_phi:.3f}",
            "p(act)": f"{entry.activation_prob:.3f}",
            "Claim": (entry.claim or "—")[:60],
        })

    st.dataframe(rows, use_container_width=True, height=350)


def render_metrics_panel(env: SotopiaEnvironment) -> None:
    st.subheader("Métricas")
    summary = env.graph.get_state_summary()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Debate Consistency", f"{env.consistency_rate:.4f}")
        st.metric("Nodos", summary["nodes"])
    with col2:
        st.metric("Densidad", f"{summary['density']:.4f}")
        st.metric("Aristas", summary["edges"])

    st.metric("Tick actual", st.session_state.current_tick)
    st.metric("Componentes", summary.get("components", 0))

    logs: list[SimulationLog] = st.session_state.all_logs
    action_counts: dict[str, int] = {}
    for l in logs:
        action_counts[l.action] = action_counts.get(l.action, 0) + 1

    parts = [f"{k}: {v}" for k, v in sorted(action_counts.items())]
    st.caption("  |  ".join(parts) if parts else "Sin acciones")

    st.caption(f"Modo: {env.orchestration_mode.value}")
    budget_info = env.budget.summary()
    if budget_info:
        st.caption("API calls: " + "  ".join(f"{k}={v}" for k, v in budget_info.items()))


def main() -> None:
    st.set_page_config(
        page_title="LangClaw Dashboard",
        page_icon="🦞",
        layout="wide",
    )

    st.title("LangClaw — Simulación Multi-Agente HRRL")
    st.caption("Regulación homeostática endógena en debate político de suma cero")

    init_session_state()

    with st.sidebar:
        st.header("Configuración")
        env_key = os.getenv("OPEN_AI_API_KEY", "")
        default_url = "https://api.openai.com/v1" if env_key else "http://localhost:11434/v1"
        default_model = "gpt-5-nano" if env_key else "llama3"

        base_url = st.text_input("API Base URL", value=default_url)
        model = st.text_input("Modelo", value=default_model)
        api_key = st.text_input("API Key", value=env_key or "ollama", type="password")
        iterations = st.slider("Iteraciones", min_value=5, max_value=100, value=50)

        orchestration_mode = st.selectbox(
            "Modo de orquestación",
            options=["hrrl", "langgraph", "round-robin", "random"],
            index=0,
            help="hrrl: regulacion homeostatica | langgraph: router LLM externo | round-robin: todos cada tick | random: agente aleatorio",
        )
        api_hard_limit = st.number_input(
            "Límite API por agente", min_value=10, max_value=1000, value=200, step=10
        )
        initial_deficit = st.slider(
            "Déficit inicial (δ₀)",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            help="Nivel de déficit epistémico al inicio. θ=0.7, λ=0.05/tick → activación 50% en tick ~(0.7-δ₀)/0.05",
        )

        st.divider()
        speed = 0.5

        if st.button("Inicializar Simulación", type="primary", use_container_width=True):
            create_environment(base_url, model, api_key, iterations, orchestration_mode, api_hard_limit, initial_deficit)
            st.rerun()

        if st.session_state.initialised:
            env = st.session_state.env
            st.success(f"Entorno listo — {iterations} ticks — modo: {env.orchestration_mode.value}")

            col_a, col_b = st.columns(2)
            with col_a:
                next_disabled = st.session_state.current_tick >= iterations
                if st.button("▶ Next Tick", disabled=next_disabled, use_container_width=True):
                    run_tick()
                    st.rerun()
            with col_b:
                if st.button("⏩ Auto-Run", use_container_width=True):
                    st.session_state.running = True
                    st.rerun()

            if st.button("⏹ Detener", use_container_width=True):
                st.session_state.running = False

            speed = st.slider("Delay entre ticks (s)", 0.1, 3.0, 0.5, 0.1)

        st.divider()
        st.caption("LangClaw v2.0.0")

    if not st.session_state.initialised:
        st.info("Configura los parámetros en la barra lateral e inicializa la simulación.")
        return

    env: SotopiaEnvironment = st.session_state.env

    if st.session_state.running and st.session_state.current_tick < iterations:
        with st.spinner(f"Ejecutando tick {st.session_state.current_tick + 1}..."):
            run_tick()
        if st.session_state.current_tick >= iterations:
            st.session_state.running = False
            st.balloons()
        else:
            time.sleep(speed)
            st.rerun()

    top_left, top_right = st.columns([3, 2])
    with top_left:
        render_graph_panel(env)
    with top_right:
        render_deficit_panel()

    mid_left, mid_right = st.columns([2, 1])
    with mid_left:
        render_utility_panel()
    with mid_right:
        render_metrics_panel(env)

    render_log_panel()


if __name__ == "__main__":
    main()
