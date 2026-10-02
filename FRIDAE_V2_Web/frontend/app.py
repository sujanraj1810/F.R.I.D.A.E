import json
import uuid
from io import BytesIO

import pandas as pd
import streamlit as st

import api

st.set_page_config(
    page_title="FRIDAE",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: #0b0f14; color: #e8edf2; }
[data-testid="stSidebar"] { background: #0e131a; border-right: 1px solid #202832; }
.fridae-title { font-size: 2.2rem; font-weight: 700; letter-spacing: .08em; margin-bottom: 0; }
.fridae-subtitle { color: #8d99a6; font-size: .9rem; }
.status-pill { display:inline-block; padding:4px 10px; border:1px solid #2b3642; border-radius:999px; font-size:.78rem; color:#aeb9c5; }
.artifact-card { border:1px solid #202832; border-radius:10px; padding:12px; margin-bottom:10px; background:#10161d; }
.muted { color:#8793a0; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

def new_chat():
    st.session_state.chat_id = f"web-{uuid.uuid4().hex}"
    st.session_state.messages = []
    st.session_state.dataset = None

if "chat_id" not in st.session_state:
    new_chat()
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

with st.sidebar:
    st.markdown("## ◈ FRIDAE")
    st.caption("Functional and Responsive Intelligent Data Analysis Engine")

    if st.button("＋ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.markdown("### Dataset")

    uploaded = st.file_uploader(
        "Upload a dataset",
        type=["csv", "json", "jsonl", "ndjson"],
        label_visibility="collapsed",
    )

    if st.button("Load Dataset", use_container_width=True, disabled=uploaded is None):
        try:
            result = api.upload(
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type or "application/octet-stream",
                st.session_state.chat_id,
            )
            st.session_state.dataset = result.get("dataset")
            st.success(f"Loaded {uploaded.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.session_state.dataset:
        ds = st.session_state.dataset
        profile = ds.get("profile") or {}
        shape = profile.get("shape") or {}
        st.markdown(f"**{ds.get('original_name', 'dataset')}**")
        st.caption(f"{shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns")

    st.divider()

    if st.button("Check Backend", use_container_width=True):
        try:
            h = api.health()
            st.success(f"Backend: {h.get('status', 'unknown')}")
        except Exception as exc:
            st.error(f"Backend unavailable: {exc}")

    if st.button("View Run Logs", use_container_width=True):
        st.session_state.show_logs = True
        st.rerun()

    if st.session_state.show_logs and st.button("Hide Logs", use_container_width=True):
        st.session_state.show_logs = False
        st.rerun()

a, b, c = st.columns([1, 2, 1])
with a:
    st.markdown('<span class="status-pill">● V2 WEB</span>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="fridae-title">FRIDAE</div>', unsafe_allow_html=True)
    st.markdown('<div class="fridae-subtitle">Your analytical AI workspace</div>', unsafe_allow_html=True)
with c:
    try:
        api.health()
        label = "BACKEND ONLINE"
    except Exception:
        label = "BACKEND OFFLINE"
    st.markdown(f'<div style="text-align:right"><span class="status-pill">{label}</span></div>',
                unsafe_allow_html=True)

chat_col, artifact_col = st.columns([2.35, 1])

with chat_col:
    st.markdown("### Conversation")

    if not st.session_state.messages:
        st.info("Ask FRIDAE to analyse data, explain a result, clean a dataset, or generate an artifact.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask FRIDAE anything…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("FRIDAE is working…"):
                try:
                    result = api.chat(prompt, st.session_state.chat_id)
                    answer = result.get("answer", "No response.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    answer = f"Request failed: {exc}"
                    st.error(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

with artifact_col:
    st.markdown("### Artifacts")

    try:
        artifacts = api.artifacts(st.session_state.chat_id).get("artifacts", [])
    except Exception:
        artifacts = []

    if not artifacts:
        st.caption("Generated files will appear here.")
    else:
        for artifact in artifacts:
            aid = artifact.get("artifact_id")
            filename = artifact.get("filename", "artifact")
            media = artifact.get("media_type", "")
            size = artifact.get("size_bytes", 0)

            st.markdown(
                f'<div class="artifact-card"><b>{filename}</b>'
                f'<br><span class="muted">{media} · {size:,} bytes</span></div>',
                unsafe_allow_html=True,
            )

            try:
                data = api.artifact_bytes(st.session_state.chat_id, aid)
                st.download_button(
                    "Download", data=data, file_name=filename,
                    mime=media or "application/octet-stream",
                    key=f"download-{aid}", use_container_width=True,
                )
                if media.startswith("image/"):
                    st.image(data, use_container_width=True)
                if media == "text/csv" or filename.lower().endswith(".csv"):
                    try:
                        st.dataframe(pd.read_csv(BytesIO(data)).head(10),
                                     use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            except Exception as exc:
                st.caption(f"Could not load artifact: {exc}")

if st.session_state.show_logs:
    st.divider()
    st.markdown("### Run Logs")
    try:
        raw = api.logs()
        rows = []
        for line in raw.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No log entries yet.")

        st.download_button(
            "Download run.jsonl", data=raw, file_name="run.jsonl",
            mime="application/x-ndjson", use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Could not load logs: {exc}")
