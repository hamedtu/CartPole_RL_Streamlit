from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st


st.set_page_config(page_title="CartPole Rollouts", layout="wide")
st.title("CartPole Rollout Viewer")

uploaded = st.file_uploader("Upload rollout_data.json", type=["json"])

data: List[Dict[str, Any]] = []
if uploaded is not None:
    try:
        data = json.load(uploaded)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

if not data:
    default_path = Path("rollout_data.json")
    if default_path.exists():
        data = json.loads(default_path.read_text())

if not data:
    st.info("Upload rollout_data.json or place it next to this app.")
    st.stop()

episode_options = [f"Episode {d.get('episode_id', i+1)} (R={d.get('total_reward', 0):.1f})" for i, d in enumerate(data)]
episode_index = st.selectbox("Select episode", options=list(range(len(data))), format_func=lambda i: episode_options[i])

episode = data[episode_index]
st.subheader(f"Episode {episode.get('episode_id', episode_index+1)}")
st.metric("Total Reward", f"{episode.get('total_reward', 0):.2f}")

steps = episode.get("steps", [])
st.write(f"Steps: {len(steps)}")

if steps:
    # Show first few transitions
    preview_rows = min(10, len(steps))
    preview = []
    for i in range(preview_rows):
        s = steps[i]
        preview.append({
            "t": i,
            "obs[0]": float(np.asarray(s.get("observation", [0,0,0,0]))[0]),
            "obs[1]": float(np.asarray(s.get("observation", [0,0,0,0]))[1]),
            "action": s.get("action"),
            "reward": s.get("reward"),
            "done": s.get("done"),
        })
    st.table(preview)

    # Plot reward per step
    rewards = [float(s.get("reward", 0.0)) for s in steps]
    st.line_chart({"reward": rewards})


