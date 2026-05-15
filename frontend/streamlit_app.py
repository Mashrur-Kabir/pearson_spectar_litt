"""
streamlit_app.py — Legal AI System Dashboard

Tabs:
  1. Documents  — Upload and manage ingested legal files
  2. Generate   — Create grounded drafts from ingested documents
  3. Edit & Learn — Submit operator edits and view learned patterns
  4. System     — Health status and configuration info
"""

from __future__ import annotations

import os
import json
from pathlib import Path
# ── Path fix — allow backend imports when run from the frontend/ directory ──
import sys
_ROOT = Path(__file__).resolve().parent.parent   # legal_ai_system/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Optional

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal AI — Pearson Specter Litt",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────

def _load_css() -> None:
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )

_load_css()

# ── API helpers ───────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs) -> Optional[dict | list]:
    """Make an API request; return parsed JSON or None on failure."""
    url = f"{API_BASE}{path}"
    try:
        resp = requests.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot connect to the backend at **{API_BASE}**. "
            "Make sure `uvicorn backend.app:app` is running."
        )
        return None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"❌ API error {exc.response.status_code}: {detail or str(exc)}")
        return None
    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")
        return None
    

@st.cache_data(ttl=30, show_spinner=False)
def _get_documents():
    return _api("GET", "/documents") or []

@st.cache_data(ttl=30, show_spinner=False)
def _get_patterns(draft_type=None):
    path = f"/patterns{'?draft_type=' + draft_type if draft_type else ''}"
    return _api("GET", path) or []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="app-header" style="flex-direction:column;align-items:flex-start;gap:4px;">
          <div class="firm-name">Pearson Specter Litt</div>
          <div class="system-name">⚖️ Legal AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:8px;'>Navigation</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Page",
        ["📂  Documents", "✍️  Generate Draft", "🔄  Edit & Learn", "🖥️  System"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.72rem;color:#475569;'>Backend: <code>{API_BASE}</code></div>",
        unsafe_allow_html=True,
    )


# ── Helper components ─────────────────────────────────────────────────────────

_TYPE_COLOURS = {
    "contract": "badge-contract",
    "notice": "badge-notice",
    "case_file": "badge-info",
    "memo": "badge-memo",
    "title_document": "badge-info",
    "unknown": "badge-warn",
}

def _badge(text: str, css_class: str = "badge-info") -> str:
    return f'<span class="badge {css_class}">{text}</span>'


def _doc_type_badge(dtype: str) -> str:
    css = _TYPE_COLOURS.get(dtype, "badge-warn")
    return _badge(dtype.replace("_", " "), css)


def _render_evidence(chunks: list) -> None:
    """Render evidence chunks as styled citation cards."""
    if not chunks:
        st.info("No evidence chunks found for this draft.")
        return
    for i, chunk in enumerate(chunks, start=1):
        score = chunk.get("score", 0)
        bar_w = int(score * 100)
        page_str = f" · p.{chunk['page_number']}" if chunk.get("page_number") else ""
        st.markdown(
            f"""
            <div class="evidence-chunk">
              <div class="source-label">[{i}] {chunk.get('filename','')}{page_str}</div>
              {chunk.get('text','')[:400]}{'…' if len(chunk.get('text','')) > 400 else ''}
              <div class="score-bar-wrap">
                <div class="score-bar" style="width:{bar_w}%"></div>
              </div>
              <div style="font-size:0.68rem;color:#475569;margin-top:4px;">
                Relevance: {score:.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📂  Documents":
    st.markdown("## 📂 Document Library")
    st.markdown(
        "<p style='color:#64748b;margin-top:-8px;'>Upload and manage ingested legal documents.</p>",
        unsafe_allow_html=True,
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    with st.expander("➕  Upload New Document", expanded=True):
        uploaded = st.file_uploader(
            "Drop a PDF or image here",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            help="Supports scanned PDFs, images, and digitally-born PDFs.",
        )
        if uploaded is not None:
            if st.button("🚀  Ingest Document", use_container_width=True):
                with st.spinner("Extracting, parsing, and indexing…"):
                    result = _api(
                        "POST", "/ingest",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    )
                if result:
                    st.cache_data.clear()
                    st.success(f"✅ {result.get('message', 'Ingestion complete.')}")
                    st.json({
                        "doc_id": result["doc_id"],
                        "pages": result["page_count"],
                        "words": result["word_count"],
                        "status": result["status"],
                    })
                    st.rerun()

    st.markdown("---")

    # ── Document list ─────────────────────────────────────────────────────────
    docs = _get_documents()
    if not docs:
        st.info("No documents ingested yet. Upload one above.")
    else:
        st.markdown(f"**{len(docs)} document(s)** in the library")
        for doc in docs:
            col_main, col_del = st.columns([10, 1])
            with col_main:
                dtype = doc.get("document_type", "unknown")
                badge_html = _doc_type_badge(dtype)
                st.markdown(
                    f"""
                    <div class="legal-card">
                      <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-weight:600;color:#e2e8f0;">{doc['filename']}</span>
                        {badge_html}
                      </div>
                      <div style="font-size:0.78rem;color:#64748b;margin-top:6px;">
                        <b style="color:#94a3b8;">ID:</b> {doc['doc_id'][:20]}…
                        &nbsp;·&nbsp;
                        <b style="color:#94a3b8;">Pages:</b> {doc.get('page_count',0)}
                        &nbsp;·&nbsp;
                        <b style="color:#94a3b8;">Words:</b> {doc.get('word_count',0):,}
                        &nbsp;·&nbsp;
                        <b style="color:#94a3b8;">Ingested:</b> {doc.get('ingested_at','')[:19]}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑", key=f"del_{doc['doc_id']}", help="Delete this document"):
                    res = _api("DELETE", f"/documents/{doc['doc_id']}")
                    if res:
                        st.cache_data.clear()
                        st.success("Deleted.")
                        st.rerun()

        # Detail expander
        with st.expander("🔍  View Full Document Metadata"):
            doc_ids = {d["filename"]: d["doc_id"] for d in docs}
            selected = st.selectbox("Select document", list(doc_ids.keys()))
            if selected:
                detail = _api("GET", f"/documents/{doc_ids[selected]}")
                if detail:
                    for field in ("parties", "dates", "case_numbers", "key_clauses"):
                        detail.pop(field + "_json", None)
                    st.json(detail)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: GENERATE DRAFT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "✍️  Generate Draft":
    st.markdown("## ✍️ Generate Grounded Draft")
    st.markdown(
        "<p style='color:#64748b;margin-top:-8px;'>Produce a legally grounded draft "
        "backed by source evidence.</p>",
        unsafe_allow_html=True,
    )

    docs = _get_documents()

    if not docs:
        st.warning("⚠️ No documents found. Go to **Documents** tab and upload one first.")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    doc_map = {d["filename"]: d["doc_id"] for d in docs}

    col_left, col_right = st.columns([3, 2])
    with col_left:
        selected_files = st.multiselect(
            "Source Documents",
            options=list(doc_map.keys()),
            help="Select one or more documents to use as source evidence.",
        )
    with col_right:
        draft_type = st.selectbox(
            "Draft Type",
            options=[
                "case_fact_summary",
                "title_review",
                "notice_summary",
                "document_checklist",
                "internal_memo",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    query = st.text_area(
        "Drafting Instruction",
        value="Generate a comprehensive legal case fact summary based on the provided documents.",
        height=90,
        help="Describe what you want the draft to contain.",
    )

    generate_btn = st.button(
        "⚡  Generate Draft",
        use_container_width=True,
        disabled=not selected_files,
    )

    if generate_btn:
        if not selected_files:
            st.error("Please select at least one source document.")
        else:
            doc_ids = [doc_map[f] for f in selected_files]
            payload = {
                "doc_ids": doc_ids,
                "query": query,
                "draft_type": draft_type,
            }
            with st.spinner("Retrieving evidence and generating draft…"):
                result = _api("POST", "/draft", json=payload)

            if result:
                st.session_state["last_draft"] = result
                st.session_state["last_draft_original"] = result.get("content", "")
                st.success(f"✅ Draft generated | ID: `{result['draft_id'][:20]}…`")

    # ── Draft viewer ──────────────────────────────────────────────────────────
    if "last_draft" in st.session_state:
        draft = st.session_state["last_draft"]
        evidence = draft.get("evidence", [])

        draft_col, evidence_col = st.columns([3, 2], gap="medium")

        with draft_col:
            st.markdown("### 📄 Draft Output")
            st.markdown(
                f'<div class="draft-output">{draft.get("content","")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:0.72rem;color:#475569;margin-top:6px;'>"
                f"Model: {draft.get('model_used','')} · "
                f"Generated: {draft.get('generated_at','')[:19]} UTC"
                f"</div>",
                unsafe_allow_html=True,
            )

        with evidence_col:
            st.markdown(f"### 🔍 Evidence ({len(evidence)} chunks)")
            _render_evidence(evidence)

    # ── Draft history ─────────────────────────────────────────────────────────
    with st.expander("📋  Draft History"):
        all_drafts = _api("GET", "/drafts") or []
        if not all_drafts:
            st.info("No drafts generated yet.")
        for d in all_drafts[:15]:
            st.markdown(
                f"""
                <div class="legal-card" style="padding:0.8rem 1rem;">
                  <span style="color:#e2e8f0;font-weight:600;">
                    {d.get('draft_type','').replace('_',' ').title()}
                  </span>
                  <span style="font-size:0.72rem;color:#64748b;margin-left:10px;">
                    {d.get('generated_at','')[:19]} UTC
                  </span>
                  <div style="font-size:0.7rem;color:#475569;margin-top:3px;">
                    ID: {d.get('draft_id','')[:32]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: EDIT & LEARN
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔄  Edit & Learn":
    st.markdown("## 🔄 Edit & Learn")
    st.markdown(
        "<p style='color:#64748b;margin-top:-8px;'>Submit your edits so the system "
        "learns your preferences for future drafts.</p>",
        unsafe_allow_html=True,
    )

    tab_edit, tab_patterns = st.tabs(["✏️  Submit Edit", "🧠  Learned Patterns"])

    # ── Tab: Submit Edit ──────────────────────────────────────────────────────
    with tab_edit:
        st.markdown("### Operator Edit Submission")
        st.markdown(
            "<p style='color:#64748b;font-size:0.85rem;'>Paste the original draft and "
            "your revised version. The system will learn from the difference.</p>",
            unsafe_allow_html=True,
        )

        # Pre-fill from last generated draft if available
        default_draft_id = ""
        default_original = ""
        if "last_draft" in st.session_state:
            default_draft_id = st.session_state["last_draft"].get("draft_id", "")
            default_original = st.session_state["last_draft"].get("content", "")

        draft_id_input = st.text_input(
            "Draft ID",
            value=default_draft_id,
            help="The draft_id returned when the draft was generated.",
        )

        col_orig, col_edit = st.columns(2)
        with col_orig:
            st.markdown("**Original Draft**")
            original_text = st.text_area(
                "Original",
                value=default_original,
                height=350,
                label_visibility="collapsed",
                key="orig_text",
            )
        with col_edit:
            st.markdown("**Your Edited Version**")
            edited_text = st.text_area(
                "Edited",
                value=default_original,
                height=350,
                label_visibility="collapsed",
                key="edited_text",
                help="Make your changes here. Differences will be analysed.",
            )

        notes = st.text_input(
            "Operator Notes (optional)",
            placeholder="e.g. 'Changed to bullet points, added missing dates.'",
        )

        if st.button("📤  Submit Edit & Learn", use_container_width=True):
            if not draft_id_input.strip():
                st.error("Please provide a Draft ID.")
            elif not original_text.strip() or not edited_text.strip():
                st.error("Both original and edited text are required.")
            elif original_text == edited_text:
                st.warning("No changes detected between original and edited versions.")
            else:
                payload = {
                    "draft_id": draft_id_input.strip(),
                    "original_content": original_text,
                    "edited_content": edited_text,
                    "operator_notes": notes or None,
                }
                with st.spinner("Analysing edits and extracting pattern…"):
                    result = _api("POST", "/edit", json=payload)
                if result:
                    st.cache_data.clear()
                    st.success("✅ Edit captured and pattern learned!")
                    st.markdown(
                        f"""
                        <div class="pattern-card">
                          <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">
                            New Pattern
                          </div>
                          <div class="pattern-desc">{result.get('description','')}</div>
                          <div class="pattern-instr">{result.get('instruction','')}</div>
                          <div style="font-size:0.7rem;color:#475569;margin-top:8px;">
                            Frequency: {result.get('frequency',1)} · 
                            ID: {result.get('pattern_id','')[:24]}…
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ── Tab: Learned Patterns ─────────────────────────────────────────────────
    with tab_patterns:
        st.markdown("### Learned Editing Patterns")
        st.markdown(
            "<p style='color:#64748b;font-size:0.85rem;'>These instructions are "
            "injected into future generation prompts automatically, ranked by frequency.</p>",
            unsafe_allow_html=True,
        )

        filter_type = st.selectbox(
            "Filter by draft type",
            ["All"] + [
                "case_fact_summary", "title_review", "notice_summary",
                "document_checklist", "internal_memo",
            ],
        )
        query_type = None if filter_type == "All" else filter_type
        patterns = _get_patterns(query_type)

        if not patterns:
            st.info("No patterns learned yet. Submit some edits to get started.")
        else:
            st.markdown(f"**{len(patterns)} pattern(s)** learned so far.")
            for pat in patterns:
                freq = pat.get("frequency", 1)
                stars = "⭐" * min(freq, 5)
                st.markdown(
                    f"""
                    <div class="pattern-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="badge badge-info">
                          {pat.get('draft_type','').replace('_',' ')}
                        </span>
                        <span style="font-size:0.75rem;color:#3b82f6;font-weight:600;">
                          {stars} ×{freq}
                        </span>
                      </div>
                      <div class="pattern-desc" style="margin-top:8px;">
                        {pat.get('description','')}
                      </div>
                      <div class="pattern-instr">{pat.get('instruction','')}</div>
                      <div style="font-size:0.68rem;color:#334155;margin-top:6px;">
                        Last applied: {pat.get('last_seen_at','')[:19]}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Recent edits table
        with st.expander("📋  Recent Operator Edits"):
            edits = _api("GET", "/edits?limit=10") or []
            if not edits:
                st.info("No edits recorded yet.")
            for e in edits:
                st.markdown(
                    f"""
                    <div class="legal-card" style="padding:0.7rem 1rem;">
                      <span style="color:#94a3b8;font-size:0.78rem;">
                        Draft: {e.get('draft_id','')[:24]}… · {e.get('captured_at','')[:19]}
                      </span>
                      <div style="font-size:0.72rem;color:#475569;margin-top:2px;">
                        {e.get('operator_notes') or 'No notes.'}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🖥️  System":
    st.markdown("## 🖥️ System Status")

    health = _api("GET", "/health")
    if health:
        status = health.get("status", "unknown")
        status_css = "badge-ok" if status == "ok" else "badge-warn"
        chroma_css = "badge-ok" if health.get("chroma_ok") else "badge-error"
        db_css = "badge-ok" if health.get("db_ok") else "badge-error"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{_badge(status.upper(), status_css)}</div>'
                f'<div class="metric-label">System</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{_badge("OK" if health.get("chroma_ok") else "FAIL", chroma_css)}</div>'
                f'<div class="metric-label">ChromaDB</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{_badge("OK" if health.get("db_ok") else "FAIL", db_css)}</div>'
                f'<div class="metric-label">SQLite</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="font-size:1.2rem;">{health.get("version","–")}</div>'
                f'<div class="metric-label">Version</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.error("Could not reach the backend health endpoint.")

    st.markdown("---")

    # Document and pattern stats
    docs = _get_documents()
    drafts = _api("GET", "/drafts") or []
    patterns = _api("GET", "/patterns") or []

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(docs)}</div>'
            f'<div class="metric-label">Documents Ingested</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(drafts)}</div>'
            f'<div class="metric-label">Drafts Generated</div></div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(patterns)}</div>'
            f'<div class="metric-label">Patterns Learned</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Configuration")
    config = _api("GET", "/config")
    if config:
        st.json(config)