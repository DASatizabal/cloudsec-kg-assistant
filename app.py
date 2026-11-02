"""CloudSec Knowledge Graph Assistant: Streamlit UI (Module 10 In-Class Activity).
Run locally:  pip install -r requirements.txt  then  streamlit run app.py
"""
import json

import streamlit as st

import kg_rag

st.set_page_config(page_title="CloudSec KG Assistant", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base and building indexes...")
def get_engine():
    return kg_rag.Engine()


engine = get_engine()

st.title("CloudSec Knowledge Graph Assistant")
st.caption("CAI 4505C · Module 10 · Graph-augmented hybrid RAG with cited, schema-validated JSON answers. "
           "Synthetic data (CVE-SIM IDs, SIM threat groups); MITRE ATT&CK technique IDs are real.")

EXAMPLES = {
    "Graph (multi-hop count)":
        "How many internet-facing prod assets have an unpatched Critical vulnerability that enables "
        "Exploit Public-Facing Application (T1190)?",
    "Runbook (time limit)":
        "What's the deadline for eradication on VMs after a brute force alert?",
    "Runbook (escalation)":
        "Who gets paged if the detection step on object storage runs late for Unsecured Credentials (T1552)?",
    "Red-team probe (poisoned RB-300)":
        "verification of Exfiltration Over Web Service (T1567) on Storage assets: what is the time limit?",
}

left, right = st.columns([2, 1])
with right:
    st.subheader("Try an example")
    choice = st.radio("Example questions", list(EXAMPLES), label_visibility="collapsed")
    st.subheader("Configuration")
    st.json(kg_rag.CONFIG, expanded=False)
    st.metric("Runbooks indexed", engine.n_docs)

with left:
    question = st.text_area("Analyst question", value=EXAMPLES[choice], height=100)
    k = st.slider("Top-k runbooks to retrieve", 1, 10, kg_rag.CONFIG["top_k"])
    if st.button("Ask", type="primary"):
        resp = engine.ask(question, k=k)
        if resp.route == "error":
            st.error(resp.answer)
        else:
            st.success(f"**Answer:** {resp.answer}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Route", resp.route)
            c2.metric("Sources cited", ", ".join(resp.sources) or "none")
            c3.metric("Latency (ms)", resp.latency_ms)
            if resp.flagged_sources:
                st.warning("Flagged and NOT used (injected instructions detected): "
                           + ", ".join(resp.flagged_sources))
            if resp.graph_facts:
                st.markdown("**Graph evidence (first 5 paths)**")
                st.code("\n".join(resp.graph_facts))
            for d in resp.sources:
                if d in engine.doc_text:
                    with st.expander(f"Cited runbook {d}"):
                        st.text(engine.doc_text[d])
        st.markdown("**Validated JSON response**")
        st.code(json.dumps(resp.model_dump(), indent=2), language="json")
