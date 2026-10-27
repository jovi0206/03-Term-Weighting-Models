"""Project 3 Streamlit: six focused pages with persistent top navigation.

M09 owns presentation/state only. Actual retrieval is delegated to M01–M08;
E8 computes optional 2-D projections without changing retrieval scores.
"""
import json
import html

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Running `streamlit run src/m09_app.py` adds src/ to import paths.
from m08_controller import SearchIndex
from project3_config import (CONDITIONS, RULES_PATH, CORPUS_PATH, DEFAULT_QUERY,
    EMBEDDING_DIR, EMBEDDING_MODELS, EMBEDDING_REPORT_PATH,
    FORMAL_CORPUS_FILE_SHA256, FORMAL_DOCUMENT_COUNT)
from e4_03_term_weighting import WEIGHTINGS
from e4_05_weighting_comparison import compare_query_document_terms, suggest_document_term_pair
from e4_07_document_term_recommendation import (recommend_document_terms,
    diagnose_document_term, term_b_on_document_change)
from e4_06_weighting_retrieval_comparison import (compare_five_method_retrieval,
    selected_pmid_after_query_change)
from e4_08_top10_presentation import build_five_top10_view
from m01_corpus_reader import sha256
from e7_01_retrieval_evaluation import load_rules, evaluate_all
from e7_04_rq4_ranking_view import make_rq4_view, RQ4_METRICS
from e7_03_weighting_interpretation import method_performance_summary
from e8_01_vector_projection import fit_document_projection, project_query
from e8_02_shared_umap import fit_shared_umap, project_shared_query
from project3_charts import (build_grouped_evaluation_chart,
    build_two_term_five_weight_chart, build_shared_umap_ranking_chart,
    shared_umap_axis_ranges)
from m04_query_processor import query_vector
from m05_similarity import score_documents
from e3_01_word2vec_viewer import (read_training_report, model_inventory,
    load_trained_word2vec, compare_embeddings)
from m01_corpus_reader import read_corpus

st.set_page_config(page_title="NCKU Project 3 | Term Weighting", page_icon=":mag:",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown('''<style>
.block-container {max-width:1250px;padding-top:2rem}
.block-container:has(.st-key-wcmp_a1_board) {max-width:1480px}
/* Project 1-inspired forest-gradient header; Project 3 content stays unchanged. */
.stApp, [data-testid="stAppViewContainer"] {background-color:#F6F7F2 !important}
[data-testid="stHeader"] {background:transparent !important}
.p3-hero {background:linear-gradient(108deg,#1C2922 0%,#294332 55%,#4B7053 100%);
    color:#FFFFFF;padding:28px 30px 27px;border-radius:17px;margin-bottom:18px;
    box-shadow:0 9px 24px rgba(25,49,32,.12)}
.p3-kicker {font-size:12px;letter-spacing:.10em;font-weight:800;color:#E4ECE5}
.p3-title {font-size:44px;line-height:1.04;font-weight:900;letter-spacing:-.02em;
    color:#FFFFFF;margin:9px 0 17px}
.p3-stages {display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.p3-stage {display:inline-block;padding:7px 12px;background:rgba(255,255,255,.10);
    border:1px solid rgba(255,255,255,.23);border-radius:999px;
    color:#FFFFFF;font-size:12px;font-weight:750;line-height:1.25}
.p3-stage-arrow {color:rgba(255,255,255,.76);font-size:15px;font-weight:700}
@media (max-width:640px) {
  .p3-hero {padding:23px 19px;border-radius:14px}
  .p3-title {font-size:37px}
  .p3-kicker {font-size:11px;line-height:1.4}
  .p3-stages {gap:5px}
  .p3-stage {font-size:11px;padding:6px 10px}
}
.p3-note {border-left:3px solid #5a8577;padding:.6rem .8rem;background:#f1f5f3;color:#203c35}
[data-testid="stSegmentedControl"] {margin-bottom:.25rem}
.wcmp-a1-method {background:rgba(35,77,66,.09);border-top:4px solid #287569;
    border-radius:7px;padding:12px 5px;text-align:center;font-size:19px;
    line-height:1.2;font-weight:800;min-height:65px}
.wcmp-rank-label {font-size:18px;font-weight:900;letter-spacing:.015em;
    padding-top:18px;text-align:center;color:#264F46;white-space:nowrap}
.wcmp-a1-card {border:1px solid #b5c7c0;border-radius:9px;
    padding:12px 11px;min-height:151px;margin-top:6px;color:#1C302A}
.wcmp-a1-card.unique {background:#DDF3E4}
.wcmp-a1-card.repeated {background:#F9DEDE;border-color:#D99F9F}
.wcmp-a1-card.selected {outline:3px solid #256c5f;outline-offset:0px}
.wcmp-a1-card .head {font-size:15px;font-weight:800;display:flex;
    justify-content:space-between;gap:6px;align-items:center;line-height:1.4}
.wcmp-a1-card .mark {background:rgba(255,255,255,.60);border-radius:6px;
    padding:3px 8px;font-size:15px;font-weight:850;color:#26332c}
.wcmp-a1-card .pmid {font-size:20px;font-weight:850;overflow-wrap:anywhere;margin:9px 0 5px}
.wcmp-a1-card .title {font-size:13px;line-height:1.4;height:38px;
    overflow-wrap:anywhere;overflow:hidden;display:-webkit-box;
    -webkit-line-clamp:2;-webkit-box-orient:vertical}
.wcmp-a1-card .score {font-size:13px;font-variant-numeric:tabular-nums;margin-top:5px}
.st-key-wcmp_a1_metrics [data-testid="stMetricLabel"] p,
.st-key-wcmp_a1_metrics [data-testid="stMetricLabel"] {
    font-size:18px!important;font-weight:800!important;line-height:1.3!important;
    color:#234b42!important;white-space:normal!important}
.st-key-wcmp_a1_metrics [data-testid="stMetricValue"] {
    font-size:36px!important;font-weight:800!important}
.wcmp-top3-legend {display:flex;gap:18px;align-items:center;flex-wrap:wrap;
    font-size:15px;font-weight:650;margin:8px 0 5px}
.wcmp-swatch {width:14px;height:14px;display:inline-block;vertical-align:middle;
    border:1px solid #c5d5ce;border-radius:3px;margin-right:6px}
@media (max-width:1450px) {
  .st-key-wcmp_a1_board {overflow-x:auto;}
  .st-key-wcmp_a1_board [data-testid="stHorizontalBlock"] {min-width:1300px;}
}
/* Five independent Top-10 tables: 3 above / 2 below; score stays visible. */
.wcmp-top10-panel {min-width:0;margin:3px 0 19px;border-radius:9px;
    background:#FFFFFF;border:1px solid #C8D5D0;overflow:hidden}
.wcmp-top10-method {background:rgba(35,77,66,.09);border-top:4px solid #287569;
    padding:11px 8px;text-align:center;color:#1E443C;font-size:20px;font-weight:850}
.wcmp-top10-table {width:100%;table-layout:fixed;border-collapse:collapse;
    color:#25352F;font-variant-numeric:tabular-nums;font-size:12.5px}
.wcmp-top10-table th {background:#E9EFEC;font-size:11.7px;font-weight:800;
    text-align:left;padding:8px 3px;white-space:nowrap}
.wcmp-top10-table td {padding:8px 3px;border-top:1px solid rgba(38,65,53,.09);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.wcmp-top10-table th:nth-child(1),.wcmp-top10-table td:nth-child(1) {width:9%;text-align:center}
.wcmp-top10-table th:nth-child(2),.wcmp-top10-table td:nth-child(2) {width:11%}
.wcmp-top10-table th:nth-child(3),.wcmp-top10-table td:nth-child(3) {width:22%}
.wcmp-top10-table th:nth-child(4),.wcmp-top10-table td:nth-child(4) {width:37%}
.wcmp-top10-table th:nth-child(5),.wcmp-top10-table td:nth-child(5) {width:21%;text-align:right;padding-right:6px}
.wcmp-top10-table tr.repeated {background:#F9DEDE}
.wcmp-top10-table tr.unique {background:#DDF3E4}
.wcmp-top10-table .score {font-weight:750;letter-spacing:-.02em}
.wcmp-top10-legend {font-size:14px;display:flex;flex-wrap:wrap;gap:18px;
    margin:5px 0 12px;color:#233D34;font-weight:650}
@media (max-width:900px) {
  .wcmp-top10-panel {overflow-x:auto}
  .wcmp-top10-table {min-width:405px}
}
.rq4-method {background:rgba(35,77,66,.09);border-top:4px solid #287569;
    border-radius:7px;padding:12px 6px;text-align:center;font-size:20px;
    line-height:1.25;font-weight:800;min-height:60px}
.rq4-rank-label {font-size:19px;font-weight:850;padding-top:20px;
    color:#264F46;white-space:nowrap;text-align:center}
.rq4-legend {display:flex;gap:18px;align-items:center;flex-wrap:wrap;
    font-size:15px;font-weight:650;margin:7px 0 12px}
.st-key-rq4_metrics [data-testid="stMetricLabel"] p,
.st-key-rq4_metrics [data-testid="stMetricLabel"] {
    font-size:17px!important;font-weight:750!important;white-space:normal!important}
.st-key-rq4_metrics [data-testid="stMetricValue"] {font-size:32px!important}
@media (max-width:890px) {
  .st-key-rq4_top3_board {overflow-x:auto}
  .st-key-rq4_top3_board [data-testid="stHorizontalBlock"] {min-width:850px}
}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner="Indexing 3,000 GLP-1 abstracts …")
def load_index(condition, corpus_hash):
    return SearchIndex.build(condition=condition)

@st.cache_data(show_spinner="Evaluating 10 queries × 5 weighting methods …")
def cached_eval(condition, corpus_hash, rules_hash):
    # Hash parameters invalidate evaluations when the underlying data/rules change.
    index=load_index(condition, corpus_hash)
    return evaluate_all(index)


@st.cache_resource(show_spinner="Projecting document vectors into 2-D …")
def cached_projection(condition, corpus_hash, method):
    # Cache the document SVD model; the user query is transformed separately.
    index = load_index(condition, corpus_hash)
    return fit_document_projection(index, method)


@st.cache_resource(show_spinner="Fitting one shared 3,000-document Cosine UMAP reference …")
def cached_shared_umap(condition, corpus_hash):
    # Only ONE fit for all five ranking overlays. Cache by corpus + preprocessing.
    return fit_shared_umap(load_index(condition, corpus_hash))


@st.cache_data(show_spinner="Projecting query onto the shared UMAP reference …")
def cached_shared_umap_query(condition, corpus_hash, query):
    # Cache by query; changing selected PMID does not refit or retransform.
    return project_shared_query(load_index(condition, corpus_hash),
                                cached_shared_umap(condition, corpus_hash), query)


@st.cache_resource(show_spinner="Loading Project 3 3,000-abstract Word2Vec model …")
def cached_word2vec(model_label, source_path, stat_size, stat_mtime_ns):
    # Stat cache key invalidates cache when model file is replaced by the user.
    return load_trained_word2vec(model_label)


def csv_bytes(rows):
    df=pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8-sig")


def section(h,subtitle):
    st.subheader(h)
    st.caption(subtitle)


def _select_weighting_document(pmid):
    """Selection is PMID-based and used by A3 two-term analysis."""
    st.session_state["wcmp_selected_pmid"] = str(pmid)
    st.session_state["wcmp_doc_source"] = "Top 10 selection"
    st.session_state.pop("wcmp_selection_expired", None)


def _a1_card(hit, marker, selected, repeated=False):
    """PMID-based Top-3 repeat classification; D markers remain stable."""
    label = html.escape(marker)
    pmid = html.escape(str(hit["pmid"]))
    full_title = str(hit["title"])
    short_title = full_title[:54] + ("…" if len(full_title) > 54 else "")
    full_title = html.escape(full_title, quote=True)
    short_title = html.escape(short_title)
    classes = "repeated" if repeated else "unique"
    if selected:
        classes += " selected"
    return (f'<div class="wcmp-a1-card {classes}">'
            f'<div class="head"><span>RANK {int(hit["rank"])}</span>'
            f'<span class="mark">{label}</span></div>'
            f'<div class="pmid">PMID {pmid}</div>'
            f'<div class="title" title="{full_title}">{short_title}</div>'
            f'<div class="score">Cosine {float(hit["score"]):.6f}</div></div>')



def _wcmp_top10_table_html(method, rows):
    """Five-column, compact per-method table; title tooltip exposes full text."""
    body = []
    for hit in rows:
        repeated_class = "repeated" if hit["Repeated"] else "unique"
        title = html.escape(str(hit["Title"]), quote=True)
        pmid = html.escape(str(hit["PMID"]))
        did = html.escape(str(hit["D ID"]))
        body.append(
            f'<tr class="{repeated_class}">'
            f'<td>{int(hit["Rank"])}</td><td><strong>{did}</strong></td>'
            f'<td title="PMID {pmid}">{pmid}</td>'
            f'<td title="{title}">{title}</td>'
            f'<td class="score">{float(hit["Cosine Score"]):.6f}</td></tr>'
        )
    if not rows:
        body.append('<tr><td colspan="5">No positive-score matches</td></tr>')
    header = "".join(f'<th scope="col">{label}</th>' for label in
                     ("Rank", "D ID", "PMID", "Title", "Cosine Score"))
    return (f'<div class="wcmp-top10-panel">'
            f'<div class="wcmp-top10-method">{html.escape(str(method))}</div>'
            f'<table class="wcmp-top10-table" aria-label="{html.escape(str(method), quote=True)} Top 10">'
            f'<thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def _rq4_select_document(pmid):
    # RQ4 selection is independent of 02/A1 selected PMID.
    st.session_state["rq4_selected_pmid"] = str(pmid)
    st.session_state["rq4_inspector_choice"] = str(pmid)


def _rq4_select_from_dropdown():
    st.session_state["rq4_selected_pmid"] = st.session_state.get("rq4_inspector_choice")


def _rq4_card(hit, marker, repeated, selected, metric_label):
    # Reuse 02's gentle two-color language; never color by score magnitude.
    full_title = str(hit.get("title", ""))
    short_title = full_title[:71] + ("…" if len(full_title) > 71 else "")
    classes = "repeated" if repeated else "unique"
    if selected:
        classes += " selected"
    return (f'<div class="wcmp-a1-card {classes}">'
            f'<div class="head"><span>RANK {int(hit["rank"])}</span>'
            f'<span class="mark">{html.escape(marker)}</span></div>'
            f'<div class="pmid">PMID {html.escape(str(hit["pmid"]))}</div>'
            f'<div class="title" title="{html.escape(full_title, quote=True)}">'
            f'{html.escape(short_title)}</div>'
            f'<div class="score">{html.escape(metric_label)}: {float(hit["score"]):.6g}'
            f'</div></div>')


def _wcmp_manual_b_edit():
    st.session_state["wcmp_b_user_edited"] = True


def _wcmp_apply_recommended_term(term):
    """One click fills the explicitly selected Term A/B input; no typing needed.

    Native Streamlit text_input does not expose browser focus to Python, so a
    visible radio target is used instead of fragile client-side JS focus hacks.
    """
    if st.session_state.get("wcmp_recommend_target", "Term B") == "Term A":
        st.session_state["wcmp_term1"] = term
    else:
        st.session_state["wcmp_term2"] = term
        st.session_state["wcmp_b_user_edited"] = True


def _wcmp_reset_to_recommended_b():
    preferred = st.session_state.get("wcmp_first_recommended_b", "")
    st.session_state["wcmp_term2"] = preferred
    st.session_state["wcmp_b_user_edited"] = False


# Persistent, top-level horizontal navigation (Project-2-style).
# st.tabs executes all tab bodies on every rerun; a controlled segmented selector
# prevents that and keeps the active page after form submissions.
MAIN_PAGES = (
    "01 Retrieval",
    "02 Weighting Comparison",
    "03 Controlled Experiments",
    "04 Evaluation",
    "05 Vector Projection",
    "06 Word2Vec Comparison",
)


def top_navigation():
    # One-time migration from the seven-page UI. Numbers 03–06 previously
    # referred to different pages; silently reusing them would navigate away
    # from the user's intended analysis after a hot reload.
    if not st.session_state.get("project3_nav_six_migrated", False):
        prior_page = st.session_state.get("project3_active_page")
        old_to_new = {
            "03 Term Weighting": "02 Weighting Comparison",
            "04 Controlled Experiments": "03 Controlled Experiments",
            "05 Evaluation": "04 Evaluation",
            "06 Vector Projection": "05 Vector Projection",
            "07 Word2Vec Comparison": "06 Word2Vec Comparison",
        }
        if prior_page in old_to_new:
            st.session_state["project3_active_page"] = old_to_new[prior_page]
        st.session_state["project3_nav_six_migrated"] = True
    if st.session_state.get("project3_active_page") not in MAIN_PAGES:
        st.session_state["project3_active_page"] = MAIN_PAGES[0]
    selector = getattr(st, "segmented_control", None)
    if callable(selector):
        active = selector("Main navigation", MAIN_PAGES,
                          key="project3_active_page",
                          selection_mode="single", label_visibility="collapsed")
    else:  # Compatibility with older Streamlit releases.
        active = st.radio("Main navigation", MAIN_PAGES, horizontal=True,
                          key="project3_active_page", label_visibility="collapsed")
    return active if active in MAIN_PAGES else MAIN_PAGES[0]


st.markdown('''<div class="p3-hero">
  <div class="p3-kicker">Artificial Intelligence Information Retrieval</div>
  <div class="p3-title">WEIGHTING</div>
  <div class="p3-stages" aria-label="Project 3 analysis workflow">
    <span class="p3-stage">3,000 Abstracts</span><span class="p3-stage-arrow" aria-hidden="true">→</span>
    <span class="p3-stage">Term Weighting</span><span class="p3-stage-arrow" aria-hidden="true">→</span>
    <span class="p3-stage">Vector Retrieval</span><span class="p3-stage-arrow" aria-hidden="true">→</span>
    <span class="p3-stage">Evaluation</span>
  </div>
</div>''', unsafe_allow_html=True)
page = top_navigation()
st.caption("Select an analysis page above. M09 displays results; M01–M08 and E modules run the calculations.")

# Corpus setting remains accessible without hiding page navigation in Sidebar.
with st.expander("Corpus / Preprocessing Settings (A–D)", expanded=False):
    condition = st.selectbox("Preprocessing (A–D)", list(CONDITIONS), index=2,
                             key="project3_preprocessing_condition",
                             format_func=lambda x: f"{x} — {CONDITIONS[x]}")
    st.caption("Fixed one-condition experiments isolate weighting effects; formal default is C.")
    st.caption("All Project 3 experiments use the same 3,000 English PMC JATS Main Abstracts. Project 2 remains unchanged.")

try:
    current_corpus_hash = sha256(CORPUS_PATH)
    if current_corpus_hash != FORMAL_CORPUS_FILE_SHA256:
        raise ValueError("Project 3 requires the verified formal 3,000-document corpus; file checksum differs")
    index = load_index(condition, current_corpus_hash)
    if index.summary()["documents"] != FORMAL_DOCUMENT_COUNT:
        raise ValueError("Formal Corpus must contain exactly 3,000 documents")
except Exception as exc:
    st.error(f"Corpus/index unavailable: {type(exc).__name__}: {exc}")
    st.stop()
summary = index.summary()
a, b, c, d = st.columns(4)
a.metric("Documents", f"{summary['documents']:,}")
b.metric("Vocabulary", f"{summary['vocabulary']:,}")
c.metric("Total tokens", f"{summary['tokens']:,}")
d.metric("Preprocessing", condition)

if page == "01 Retrieval":
    section("Vector Space Search", "Same vocabulary and weighting formula for query and documents; rank by cosine similarity.")
    with st.form("search_form"):
        query=st.text_input("Query",value=st.session_state.get("last_query","semaglutide weight loss"))
        col1,col2=st.columns(2)
        method=col1.selectbox("Weighting method", WEIGHTINGS, index=3)
        metric=col2.selectbox("Scoring",["Cosine","Raw Dot Product","L2 Dot Product"])
        submit=st.form_submit_button("Search Top 10",type="primary")
    if (submit or "last_search" not in st.session_state
            or st.session_state.get("last_condition") != condition
            or st.session_state.get("last_search_corpus_hash") != index.corpus_sha256):
        st.session_state["last_query"]=query
        st.session_state["last_condition"]=condition
        st.session_state["last_search_corpus_hash"] = index.corpus_sha256
        st.session_state["last_search"]=index.search(query,method,metric,limit=10)
    results=st.session_state["last_search"]
    if results["oov"]:
        st.warning("Out of vocabulary: " + ", ".join(results["oov"]))
    if not results['query_has_weights']:
        st.info("Query contains no nonzero-weight in-vocabulary terms for this weighting.")
    st.caption(f"Matching documents: {results['nonzero_documents']} · Query L2 norm: {results['query_l2_norm']:.4f} · {results['method']} / {results['metric']}")
    for hit in results["hits"]:
        st.markdown(f"**{hit['rank']:02d}. {hit['title']}**")
        st.caption(f"PMID: {hit['pmid']} · PMCID: {hit['pmcid']} · Score: {hit['score']:.6f} · Length: {hit['length']} terms")
        with st.expander("Read main Abstract / term weights"):
            st.write(hit['abstract'])
            st.caption("Term weights shown for query terms and this retrieved document (all non-query terms may also appear).")
            rows=[r for r in index.explain_terms(results['query'],hit['doc_index'],results['method'])
                  if r['Query TF'] or r['Document TF']>2][:120]
            st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.download_button("Export Top 10 CSV",csv_bytes(results['hits']),"project3_search_top10.csv","text/csv")

elif page == "02 Weighting Comparison":
    section("Weighting Comparison",
            "A1: five-method ranking · A2: shared UMAP · A3: two terms × five weighting methods.")
    st.caption(f"3,000 JATS Main Abstracts · Preprocessing {condition} (baseline C) "
               "· five weightings · original high-dimensional Cosine · Top 10")
    with st.expander("Experiment notes / limitations", expanded=False):
        st.markdown("All five methods use the same Query, Corpus, Preprocessing and Cosine "
                    "retrieval procedure; only term weighting changes. "
                    "**Cosine scores from different weighted vector spaces must not "
                    "be compared directly by magnitude.** "
                    "UMAP is a shared 2D visual reference, not a ranking metric. "
                    "A3 shows term weights rather than normalized cosine scores; "
                    "ranking differences alone do not prove higher relevance quality "
                    "(see 04 Evaluation / Proxy Qrels).")
    st.markdown("### A1 · Five Methods Retrieval Comparison")
    # Isolated versioned keys migrate the previous default without affecting 01/04/06.
    if "wcmp_active_query_v2" not in st.session_state:
        st.session_state["wcmp_active_query_v2"] = "glp-1"
    with st.form("wcmp_a1_query_form"):
        entered_query = st.text_input("One query for all five weighting methods",
            value=st.session_state["wcmp_active_query_v2"], key="wcmp_query_v2")
        compare_submitted = st.form_submit_button("Compare 5 methods · Top 10", type="primary")
    if compare_submitted:
        st.session_state["wcmp_active_query_v2"] = entered_query.strip()
    comparison_query = st.session_state["wcmp_active_query_v2"]
    if not comparison_query:
        st.info("Enter a query above to compare real rankings; no placeholder results are generated.")

    # E4 only calls the SAME M08.search / M06 ranking exposed by 01 Retrieval.
    a1 = compare_five_method_retrieval(index, comparison_query)
    ranked = a1["rankings"]
    pmid_to_index = a1["pmid_to_index"]
    markers = a1["labels"]
    stats = a1["summary"]
    first_result = ranked[WEIGHTINGS[0]]
    if first_result["oov"]:
        st.warning("Query OOV terms (shared across all methods): " +
                   ", ".join(first_result["oov"]))
    if not any(r["hits"] for r in ranked.values()):
        st.warning("No positive-cosine matches for this query under the five methods. "
                   "Choose a different query or manually enter a PMID below.")
    if st.session_state.get("wcmp_doc_source") not in ("Top 10 selection", "PMID"):
        st.session_state["wcmp_doc_source"] = "Top 10 selection"
    source = st.radio("Selected document source", ["Top 10 selection", "PMID"],
                      horizontal=True, key="wcmp_doc_source")

    context = (index.corpus_sha256, condition, comparison_query)
    previous_context = st.session_state.get("wcmp_previous_a1_context")
    context_changed = previous_context is not None and previous_context != context
    selected_pmid, expired = selected_pmid_after_query_change(
        st.session_state.get("wcmp_selected_pmid"), source, pmid_to_index, context_changed)
    st.session_state["wcmp_previous_a1_context"] = context
    if source == "Top 10 selection" and selected_pmid is not None and selected_pmid not in pmid_to_index:
        selected_pmid, expired = None, True
    if expired:
        st.session_state["wcmp_selected_pmid"] = None
        st.session_state["wcmp_selection_expired"] = True
    if st.session_state.get("wcmp_selection_expired") and source == "Top 10 selection":
        st.warning("Previously selected PMID is not in the current five-method Top 10 pool. "
                   "The selection has been cleared. Choose a new result or use PMID mode "
                   "to inspect that article explicitly.")

    st.markdown("**Five methods · aligned Top 3 (Select a PMID to inspect A3)**")
    top3_repeated = set(stats["top3_repeated_pmids"])
    st.markdown('''<div class="wcmp-top3-legend">
        <span><i class="wcmp-swatch" style="background:#F9DEDE"></i>Repeated PMID in ≥2 Top 3 lists</span>
        <span><i class="wcmp-swatch" style="background:#DDF3E4"></i>Unique to one Top 3 list</span>
        </div>''', unsafe_allow_html=True)
    try:
        board = st.container(key="wcmp_a1_board")
    except TypeError:
        board = st.container()
    with board:
        headers = st.columns([0.78, 1, 1, 1, 1, 1], gap="small")
        with headers[0]:
            st.markdown("**RANK**")
        for method_pos, method in enumerate(WEIGHTINGS):
            with headers[method_pos + 1]:
                st.markdown(f'<div class="wcmp-a1-method">{html.escape(method)}</div>',
                            unsafe_allow_html=True)
        # Each actual rank occupies one SHARED ROW across five methods.
        for rank in range(1, 4):
            line = st.columns([0.78, 1, 1, 1, 1, 1], gap="small")
            with line[0]:
                st.markdown(f'<div class="wcmp-rank-label">RANK {rank}</div>',
                            unsafe_allow_html=True)
            for method_pos, method in enumerate(WEIGHTINGS):
                with line[method_pos + 1]:
                    if rank > len(ranked[method]["hits"]):
                        st.caption("No positive-score document")
                        continue
                    hit = ranked[method]["hits"][rank - 1]
                    pmid = str(hit["pmid"])
                    label = markers[pmid]
                    st.markdown(_a1_card(hit, label,
                        source == "Top 10 selection" and selected_pmid == pmid,
                        repeated=pmid in top3_repeated), unsafe_allow_html=True)
                    st.button(f"Select {label}",
                        key=f"wcmp_a1_select_{method_pos}_{rank}_{pmid}",
                        on_click=_select_weighting_document, args=(pmid,),
                        use_container_width=True,
                        type="primary" if source == "Top 10 selection" and selected_pmid == pmid else "secondary")
    try:
        metric_area = st.container(key="wcmp_a1_metrics")
    except TypeError:
        metric_area = st.container()
    with metric_area:
        metric_cols = st.columns(3, gap="medium")
        agreement = stats["top1_max_agreement"]
        metric_cols[0].metric("Top 1 Agreement", f"{agreement} of 5 methods")
        metric_cols[1].metric("Repeated PMIDs in Top 3", len(stats["top3_repeated_pmids"]))
        metric_cols[2].metric("Unique PMIDs across Top 10", stats["unique_top10_pmids"])
    if agreement > 1:
        for group in stats["top1_agreement_groups"]:
            st.caption(f"{', '.join(group['methods'])} share Top 1 PMID {group['pmid']} "
                       f"({markers[group['pmid']]}).")
    elif agreement == 1:
        st.caption("All available methods have different Top 1 PMIDs.")
    else:
        st.caption("No method returned a positive-score Top 1 document.")
    st.caption(f"Unique Top 1 PMIDs: {stats['top1_unique_pmids']}")
    repeated = stats["top3_repeated_pmids"]
    if repeated:
        st.caption("Repeated in at least two Top 3 lists: " + ", ".join(
            f"{markers[pmid]} (PMID {pmid})" for pmid in repeated))
    st.caption(f"Top 3 distinct PMIDs: {stats['unique_top3_pmids']} · "
               f"Shared by all five Top 3 lists: {len(stats['top3_shared_by_all_five'])}")
    st.markdown("### A2 · Five Methods UMAP Projection")
    st.caption("Shared Log TF-IDF Cosine UMAP · fixed document coordinates across five methods "
               "· orange: method Top 10 · red star: Query")
    try:
        projection = cached_shared_umap(condition, index.corpus_sha256)
        query_xy, query_warning = cached_shared_umap_query(
            condition, index.corpus_sha256, comparison_query)
        if query_warning:
            st.warning(query_warning)
        if projection.parameters["plotted_documents"] < projection.parameters["corpus_documents"]:
            st.warning(f"{projection.parameters['corpus_documents'] - projection.parameters['plotted_documents']} "
                       "zero-weight reference documents have no valid Cosine location and are not plotted.")
        import numpy as np
        valid_xy = projection.coords[np.isfinite(projection.coords).all(axis=1)]
        if query_xy is not None:
            valid_xy = np.vstack((valid_xy, query_xy))
        axes = shared_umap_axis_ranges(valid_xy)
        # 2 + 2 + 1 centered; plots share the precise same document/query coordinates.
        for pair in (WEIGHTINGS[0:2], WEIGHTINGS[2:4]):
            cols = st.columns(2, gap="medium")
            for col, method in zip(cols, pair):
                with col:
                    st.plotly_chart(build_shared_umap_ranking_chart(
                        index, projection, method, ranked[method], markers,
                        query_xy, axes), width="stretch",
                        key=f"wcmp_umap_{WEIGHTINGS.index(method)}")
        final_cols = st.columns([1, 2, 1], gap="medium")
        with final_cols[1]:
            method = WEIGHTINGS[4]
            st.plotly_chart(build_shared_umap_ranking_chart(
                index, projection, method, ranked[method], markers,
                query_xy, axes), width="stretch", key="wcmp_umap_4")
    except (ValueError, RuntimeError, ImportError) as exc:
        st.error(f"Shared UMAP visualization unavailable: {exc}")
        st.caption("Retrieval and A3 remain functional; this projection does not affect ranking.")

    with st.expander("Full Top 10 · five independent weighting tables (50 ranking positions)", expanded=False):
        top10_view = build_five_top10_view(ranked, markers)
        st.markdown('''<div class="wcmp-top10-legend">
            <span><i class="wcmp-swatch" style="background:#F9DEDE"></i>Repeated PMID in ≥2 Top 10 lists</span>
            <span><i class="wcmp-swatch" style="background:#DDF3E4"></i>Unique to one Top 10 list</span>
            </div>''', unsafe_allow_html=True)
        # Same M08 search hits and D labels as A1; only the presentation changes.
        # First row: 3 methods; second row: 2 methods (centered, equal widths).
        with st.container(key="wcmp_top10_board"):
            first_row = st.columns(3, gap="medium")
            for panel, method in zip(first_row, WEIGHTINGS[:3]):
                with panel:
                    st.markdown(_wcmp_top10_table_html(
                        method, top10_view["tables"][method]), unsafe_allow_html=True)
            second_row = st.columns([1, 2, 2, 1], gap="medium")
            for panel, method in zip(second_row[1:3], WEIGHTINGS[3:]):
                with panel:
                    st.markdown(_wcmp_top10_table_html(
                        method, top10_view["tables"][method]), unsafe_allow_html=True)
        st.caption("All five score columns are original high-dimensional Cosine values. "
                   "They are NOT directly comparable across weighting spaces. "
                   "Hover over Title for the complete title.")
        st.download_button("Export five methods Top 10 CSV",
            csv_bytes(top10_view["csv_rows"]),
            "project3_weighting_five_top10.csv", "text/csv")
        if pmid_to_index:
            pooled_pmid = st.selectbox("Choose any PMID from the full five-method Top 10 pool",
                list(pmid_to_index), key="wcmp_full_top10_pmid",
                format_func=lambda pmid: f"{markers[pmid]} · PMID {pmid} · {index.docs[pmid_to_index[pmid]]['title'][:85]}")
            st.button("Analyze selected Top 10 PMID in A3", key="wcmp_full_top10_choose",
                on_click=_select_weighting_document, args=(pooled_pmid,))
        st.caption("Only actual positive-score retrieval hits are shown; uncommon/OOV queries "
                   "can return fewer than 10 documents per method.")
    with st.expander("Ranking overlap · pairwise Top 3 / Top 10", expanded=False):
        st.dataframe(pd.DataFrame(stats["pairwise_overlap"]), hide_index=True, width="stretch")
    st.markdown("---")
    st.markdown("### A3 · Two Terms × Five Methods")
    chosen_doc = None
    if source == "Top 10 selection":
        if selected_pmid in pmid_to_index:
            chosen_doc = pmid_to_index[selected_pmid]
            st.success(f"Selected Document: PMID {selected_pmid} (from A1)")
        else:
            st.info("Choose any **Select Dxx** button in A1 to inspect that document below. "
                    "No replacement PMID is selected automatically.")
    else:
        entered_pmid = st.text_input("Enter PMID (exact match)",
            value=st.session_state.get("wcmp_selected_pmid") or "35503498",
            key="wcmp_pmid").strip()
        chosen_doc = next((i for i, doc in enumerate(index.docs)
                           if str(doc["pmid"]) == entered_pmid), None)
        if chosen_doc is None:
            st.warning("PMID not found in the formal 3,000-document corpus.")
        else:
            st.success(f"Selected Document: PMID {entered_pmid} (manual mode)")
    # A3 term recommendations use the same selected document/index as the
    # term statistics. No separate corpus, fallback PMID, or second tokenizer.
    # One-time UI migration: the new default A must be glp-1 even in an old
    # session that still contains a prior manually entered Term A. Subsequent
    # reruns and PMID changes retain the user's current A/B selections.
    if "wcmp_defaults_a3a4_v3" not in st.session_state:
        st.session_state["wcmp_term1"] = "glp-1"
        st.session_state["wcmp_b_user_edited"] = False
        st.session_state.pop("wcmp_b_document_context", None)
        st.session_state["wcmp_recommend_target"] = "Term B"
        st.session_state["wcmp_defaults_a3a4_v3"] = 1
    current_term_a = str(st.session_state.get("wcmp_term1", "glp-1"))
    candidates = (recommend_document_terms(index, chosen_doc, term_a=current_term_a)
                  if chosen_doc is not None else [])
    recommended_terms = [row["Term"] for row in candidates]
    first_recommended = recommended_terms[0] if recommended_terms else ""
    st.session_state["wcmp_first_recommended_b"] = first_recommended
    selected_doc_key = (index.corpus_sha256, condition,
                        str(index.docs[chosen_doc]["pmid"]) if chosen_doc is not None else None,
                        current_term_a)
    if "wcmp_b_user_edited" not in st.session_state:
        st.session_state["wcmp_b_user_edited"] = False
    if st.session_state.get("wcmp_b_document_context") != selected_doc_key:
        # Document/Term A changes update B only in automatic mode.
        # Manually edited text/candidate selection must not be overridden.
        st.session_state["wcmp_term2"] = term_b_on_document_change(
            st.session_state.get("wcmp_b_document_context"), selected_doc_key,
            st.session_state.get("wcmp_term2", ""),
            st.session_state["wcmp_b_user_edited"], first_recommended)
        st.session_state["wcmp_b_document_context"] = selected_doc_key
    mode = st.radio("Comparison mode", ["Two terms", "Single term"],
                    horizontal=True, key="wcmp_mode")
    one, two = st.columns(2)
    term1 = one.text_input("Term A", key="wcmp_term1")
    if mode == "Two terms":
        if "wcmp_term2" not in st.session_state:
            st.session_state["wcmp_term2"] = first_recommended
        term2 = two.text_input("Term B", key="wcmp_term2",
                              on_change=_wcmp_manual_b_edit)
    else:
        term2 = None

    if candidates:
        st.markdown("**Recommended Terms from Selected Document**")
        st.caption("Top 5 terms from this PMID's Main Abstract (TF > 0). "
                   "Statistics/medical-term rules, not semantic AI. "
                   "Choose the destination, then click a term to fill it without typing.")
        target_options = ["Term A", "Term B"] if mode == "Two terms" else ["Term A"]
        if st.session_state.get("wcmp_recommend_target") not in target_options:
            st.session_state["wcmp_recommend_target"] = "Term A"
        st.radio("Fill recommended term into", target_options, horizontal=True,
                 key="wcmp_recommend_target")
        st.dataframe(pd.DataFrame(candidates)[
            ["Term", "Document TF", "Corpus DF", "IDF (log10)",
             "Higher IDF than Term A"]], hide_index=True, width="stretch")
        term_buttons = st.columns(3)
        for n, row in enumerate(candidates):
            with term_buttons[n % 3]:
                st.button(row["Term"], key=f"wcmp_apply_recommended_{n}",
                          on_click=_wcmp_apply_recommended_term,
                          args=(row["Term"],), use_container_width=True,
                          help=f"Fill selected field · Document TF {row['Document TF']} "
                               f"· IDF {row['IDF (log10)']:.4f}")
        if mode == "Two terms":
            if st.session_state["wcmp_b_user_edited"]:
                st.caption("Manually selected Term B is preserved across PMID changes. "
                           "Use automatic Term B to follow the selected document again.")
            st.button("Resume automatic Term B", key="wcmp_auto_b",
                      on_click=_wcmp_reset_to_recommended_b)
    elif chosen_doc is not None:
        st.info("No eligible document terms for recommendation. "
                "Enter Term A/B manually.")

    if chosen_doc is not None:
        d = index.docs[chosen_doc]
        st.caption(f"Selected document · PMID **{d['pmid']}** · {d['title']} · "
                   f"Preprocessing {condition} · Query: **{comparison_query}**")
        try:
            term_summaries, all_weights = compare_query_document_terms(
                index, comparison_query, chosen_doc,
                [term1, term2] if term2 is not None else [term1])
            st.markdown("#### Exact source values · TF / CF / DF / IDF")
            st.dataframe(pd.DataFrame(term_summaries), hide_index=True, width="stretch")
            st.markdown("#### Five methods · exact Query / Document term weights")
            st.dataframe(pd.DataFrame(all_weights), hide_index=True, width="stretch")
            st.download_button("Export weighting comparison CSV", csv_bytes(all_weights),
                               "project3_weighting_comparison.csv", "text/csv")
            for row in term_summaries:
                term = row["Term"]
                if row["IDF (log10)"] is None:
                    st.warning(f"'{term}' is OOV for preprocessing {condition}; IDF is undefined "
                               "and query/document weights are zero.")
                if row["Query TF"] == 0:
                    st.info(f"Selected term **{term}** is absent from the query; all its query weights are zero.")
                if row["Document TF"] == 0:
                    detail = diagnose_document_term(index, chosen_doc, term)
                    st.warning(f"PMID {detail.get('pmid', 'N/A')} · **{term}**: "
                               f"Document TF = 0. {detail['reason']} "
                               "All document weights for this term are therefore zero.")
            if len(term_summaries) == 2:
                st.markdown("#### A3 · Two terms × five methods · same document")
                st.plotly_chart(build_two_term_five_weight_chart(all_weights), width="stretch")
                if any(row["Document TF"] == 0 for row in term_summaries):
                    st.caption("A3 includes a term absent from the selected Main Abstract. "
                               "Choose an in-document recommendation for a nonzero comparison.")
            else:
                st.caption("A3 compares TWO terms. Select 'Two terms' above to display the chart; "
                           "single-term source values remain available in the table and CSV.")
        except (ValueError, IndexError) as exc:
            st.warning(f"Weighting comparison input: {exc}")

elif page == "03 Controlled Experiments":
    section("Controlled Experiments",
            "Fixed query, weighting, corpus and preprocessing; compare three similarity metrics on the SAME document vectors.")
    # RQ4's control condition is C regardless of another page's exploratory A–D choice.
    rq4_index = index if condition == "C" else load_index("C", current_corpus_hash)
    st.caption("Fixed: 3,000 GLP-1 JATS Main Abstracts · Preprocessing C · Top 10 "
               "· one weighting for all three metrics")
    inputs = st.columns([1.4, 1])
    query = inputs[0].text_input("Query · all three metrics", value="glp-1", key="rq4_query_v2")
    weighting = inputs[1].selectbox("Fixed weighting", WEIGHTINGS,
                                   index=WEIGHTINGS.index("Raw TF-IDF"), key="rq4_weighting_v2")
    if not query.strip():
        st.warning("Enter a query to run RQ4. No placeholder rankings will be shown.")
    else:
        rq4 = make_rq4_view(rq4_index, query.strip(), weighting, k=10)
        if rq4["oov"]:
            st.warning("OOV query terms: " + ", ".join(rq4["oov"]))
        if rq4["no_positive_hits"]:
            st.warning("No positive-score matches: try another in-vocabulary query.")

        # Clear stale 03 selection on query/weighting/corpus change only.
        rq4_context = (rq4_index.corpus_sha256, "C", query.strip(), weighting)
        if st.session_state.get("rq4_previous_context") != rq4_context:
            st.session_state["rq4_selected_pmid"] = None
            st.session_state.pop("rq4_inspector_choice", None)
            st.session_state["rq4_previous_context"] = rq4_context
        selected_rq4_pmid = st.session_state.get("rq4_selected_pmid")

        st.markdown("### Three Similarity Methods · Aligned Top 3")
        st.markdown('''<div class="rq4-legend">
            <span><i class="wcmp-swatch" style="background:#F9DEDE"></i>Repeated PMID across methods</span>
            <span><i class="wcmp-swatch" style="background:#DDF3E4"></i>Unique PMID</span>
            </div>''', unsafe_allow_html=True)
        try:
            rq4_board = st.container(key="rq4_top3_board")
        except TypeError:
            rq4_board = st.container()
        with rq4_board:
            heads = st.columns([0.75, 1.25, 1.25, 1.25], gap="small")
            heads[0].markdown("**RANK**")
            for column, (_, label) in zip(heads[1:], RQ4_METRICS):
                column.markdown(f'<div class="rq4-method">{html.escape(label)}</div>',
                                unsafe_allow_html=True)
            for rank in range(1, 4):
                columns = st.columns([0.75, 1.25, 1.25, 1.25], gap="small")
                columns[0].markdown(f'<div class="rq4-rank-label">RANK {rank}</div>',
                                    unsafe_allow_html=True)
                for column, (metric, short_name) in zip(columns[1:], RQ4_METRICS):
                    hits = rq4["rankings"][metric]["hits"]
                    if rank > len(hits):
                        column.caption("No positive-score document")
                        continue
                    hit = hits[rank - 1]
                    pmid = str(hit["pmid"])
                    marker = rq4["markers"][pmid]
                    with column:
                        st.markdown(_rq4_card(hit, marker,
                            pmid in rq4["top3_repeated_pmids"],
                            selected_rq4_pmid == pmid, short_name),
                            unsafe_allow_html=True)
                        st.button(f"Inspect {marker}", key=f"rq4_pick_{metric}_{rank}_{pmid}",
                                  on_click=_rq4_select_document, args=(pmid,),
                                  use_container_width=True,
                                  type="primary" if selected_rq4_pmid == pmid else "secondary")

        try:
            metric_area = st.container(key="rq4_metrics")
        except TypeError:
            metric_area = st.container()
        with metric_area:
            cols = st.columns(2, gap="medium")
            cols[0].metric("Raw Dot vs Cosine · Top 10 Overlap",
                           f'{rq4["raw_cos_top10_overlap"]}/10')
            cols[1].metric("Cosine vs L2 Dot · Same Ordered Top 10",
                           str(rq4["cos_l2_same_order"]))
        if not rq4["cos_l2_same_order"] and not rq4["no_positive_hits"]:
            st.warning("Cosine and L2 Dot rankings differ. Inspect possible numerical "
                       "ties or zero-vector edge cases before interpreting the result.")
        # RQ4 technical/mathematical notes remain in README_RQ4_FINALUI.md;
        # the main UI focuses on aligned rankings and the two requested metrics.

        with st.expander("Full Top 10 · Raw Dot / Cosine / L2 Dot", expanded=False):
            cols = st.columns(3, gap="small")
            for column, (metric, short_name) in zip(cols, RQ4_METRICS):
                with column:
                    st.markdown(f"**{short_name}**")
                    table = pd.DataFrame(rq4["tables"][metric])
                    if table.empty:
                        st.caption("No matches")
                        continue
                    # Highlight using Raw Dot / Cosine Top-10 membership only.
                    styled = table[["Rank", "D ID", "PMID", "Title", "Score"]].style.apply(
                        lambda row: ["background-color: #F9DEDE"
                                     if str(row["PMID"]) in rq4["top10_repeated_pmids"]
                                     else "background-color: #DDF3E4"] * len(row), axis=1)
                    st.dataframe(styled, hide_index=True, width="stretch", height=410)
            st.caption("Pink = PMID in both Raw Dot and Cosine Top 10; "
                       "green = not in both. L2 Dot duplicates are not an extra vote.")

        # Article inspection is page-local and supports any Top-10 result.
        options = [None] + list(rq4["markers"])
        current = st.session_state.get("rq4_selected_pmid")
        st.selectbox("Inspect any Top 10 article", options,
            index=options.index(current) if current in options else 0,
            format_func=lambda pmid: ("Choose a document" if pmid is None else
                f'{rq4["markers"][pmid]} · PMID {pmid} · '
                f'{str(rq4["hit_by_pmid"][pmid].get("title", ""))[:80]}'),
            key="rq4_inspector_choice", on_change=_rq4_select_from_dropdown)
        current = st.session_state.get("rq4_selected_pmid")
        if current in rq4["hit_by_pmid"]:
            hit = rq4["hit_by_pmid"][current]
            st.markdown(f'#### {rq4["markers"][current]} · PMID {html.escape(current)}')
            st.markdown(f'**{html.escape(str(hit.get("title", "")))}**')
            st.caption(f'PMCID: {hit.get("pmcid", "N/A")} · '
                       f'Document length: {hit.get("length", "N/A")} tokens · '
                       f'L2 norm under {weighting}: {float(hit.get("l2_norm", 0)):.6g}')
            for metric, short_name in RQ4_METRICS:
                matching = next((h for h in rq4["rankings"][metric]["hits"]
                                 if str(h["pmid"]) == current), None)
                if matching is None:
                    st.caption(f"{short_name}: outside Top 10")
                else:
                    st.caption(f'{short_name}: Rank {matching["rank"]} · '
                               f'Score {float(matching["score"]):.10g}')
            with st.expander("Read full Main Abstract", expanded=True):
                st.write(hit.get("abstract", "No Main Abstract text available."))

        st.download_button("Export RQ4 Top 10 CSV", csv_bytes(rq4["export_rows"]),
                           "project3_rq4_similarity_top10.csv", "text/csv")

elif page == "04 Evaluation":
    section("Retrieval Evaluation", "10 thematic queries, explicit transparent query relevance rules, four IR metrics at k=10.")
    rows, means, qrels = cached_eval(condition, index.corpus_sha256, sha256(RULES_PATH))
    st.warning("The uploaded corpus has no independently judged relevance labels. These qrels are automated abstract-topic regex PROXIES, not human-reviewed ground truth. Reported metrics are only exploratory and favor lexical topic matching.")
    st.markdown("**Mean performance across all 10 queries**")
    st.dataframe(pd.DataFrame(means).style.format({k:"{:.4f}" for k in ["Precision@10","Recall@10","MRR@10","nDCG@10"]}),hide_index=True,width="stretch")
    # Presentation-only grouped bars; never sum different IR metrics.
    # Uses the same ``means`` records shown in the table and CSV export.
    st.plotly_chart(build_grouped_evaluation_chart(means), width="stretch")
    st.caption("Grouped, not stacked: five weighting methods within each metric. "
               "Scores range from 0 to 1; do not sum different metrics or interpret "
               "one method as universally best.")

    # Explain the Top-10 ceiling using the EXISTING proxy-qrels denominator;
    # neither relevance labels nor precision/recall/evaluation calculations change.
    rules = load_rules()
    with st.expander("Why is Recall@10 relatively low?", expanded=False):
        st.write("Recall@10 is the proportion of ALL proxy-labeled relevant documents "
                 "retrieved within the first ten results. When many relevant papers "
                 "exist, returning only ten limits the maximum achievable recall.")
        example_rule = next((rule for rule in rules
                             if rule['query'].strip().casefold() == 'semaglutide weight loss'), None)
        if example_rule is not None:
            relevant_count = len(qrels[example_rule['id']])
            if relevant_count:
                ceiling = min(10, relevant_count) / relevant_count
                st.markdown(f"For **`semaglutide weight loss`**, the current Proxy Qrels "
                            f"contain **{relevant_count:,}** relevant documents. "
                            f"Even if all Top 10 results were relevant, the maximum "
                            f"Recall@10 would be **{min(10, relevant_count)} / "
                            f"{relevant_count:,} = {ceiling:.4f}**.")
        st.caption("Proxy Qrels are abstract-regex labels, NOT independently judged "
                   "ground truth. Low recall alone does not prove poor retrieval.")
    with st.expander("How to interpret the four IR metrics", expanded=False):
        st.markdown("**Precision@10:** fraction of the Top 10 documents that are relevant.  "
                    "\n**Recall@10:** fraction of all relevant documents found in the Top 10.  "
                    "\n**MRR@10:** reciprocal rank of the first relevant document (0 if absent).  "
                    "\n**nDCG@10:** rank-discounted quality of the Top 10 ordering.  "
                    "\nHigher is generally better within each metric; the four "
                    "metrics measure different things and must not be added.")

    st.markdown("#### Teacher follow-up · Why can TF-IDF be useful without always winning?")
    evidence=method_performance_summary(means)
    st.dataframe(pd.DataFrame(evidence),hide_index=True,width="stretch")
    by_method={row["Method"]:row for row in evidence}
    raw=by_method["Raw TF-IDF"]
    log=by_method["Log TF-IDF"]
    st.caption(f"Under the current {condition} preprocessing and 10 proxy-labeled queries: "
               f"Raw TF-IDF is rank {raw['nDCG rank (of 5)']}/5 in mean nDCG@10 "
               f"({raw['nDCG@10']:.4f}); Log TF-IDF is rank {log['nDCG rank (of 5)']}/5 "
               f"({log['nDCG@10']:.4f}). These are experiment-specific ranks, not universal claims.")
    st.info("**Separate three meanings of 'low rank/score':** (1) an individual term's "
            "TF-IDF weight, (2) a document's Cosine score for one query, and "
            "(3) an algorithm's retrieval quality such as mean nDCG@10 are different quantities. "
            "Even a small absolute Cosine score can be ranked #1 if it is the highest among documents. "
            "TF-IDF is useful because it is training-free, sparse, interpretable, quick to recompute, "
            "and a strong lexical baseline. It is not universally best: semantic paraphrases, "
            "length effects, corpus composition and query types may favor other methods. "
            "Our proxy qrels are lexically derived, so these quality numbers are exploratory.")
    st.markdown("**Per-query evaluation**")
    st.dataframe(pd.DataFrame(rows).drop(columns=["Relevant PMIDs"]),hide_index=True,width="stretch")
    choice=st.selectbox("Relevance judgment rule",[r['id'] for r in rules],
                        format_func=lambda rid:next(r['query'] for r in rules if r['id']==rid))
    rule=next(r for r in rules if r['id']==choice)
    st.write(rule['topic'])
    st.json({"Query":rule['query'],"AND concepts / OR regex variants":rule['all_concepts'],
             "Matching PMID count":len(qrels[choice])})
    with st.expander("Inspect proxy-labeled relevant abstracts"):
        for pos,i in enumerate(sorted(qrels[choice])[:15],1):
            doc=index.docs[i]
            st.markdown(f"{pos}. **{doc['title']}** (PMID: {doc['pmid']})")
            st.caption(doc['abstract'][:650])
        st.caption("First 15 in source order; full PMID set in proxy_qrels.json.")
    st.download_button("Export per-query CSV",csv_bytes(rows),"project3_per_query_evaluation.csv","text/csv")
    st.download_button("Export mean CSV",csv_bytes(means),"project3_evaluation_mean.csv","text/csv")
    export={rid:sorted(index.docs[i]['pmid'] for i in indices) for rid,indices in qrels.items()}
    st.download_button("Export proxy qrels JSON",json.dumps(export,ensure_ascii=False,indent=2).encode("utf-8"),"project3_proxy_qrels.json","application/json")
    st.caption("Metrics: P@10 divides by 10; Recall uses all rule-labeled relevant docs; MRR=first relevant reciprocal rank; nDCG@10 uses binary relevance. Rankings always use cosine.")

elif page == "05 Vector Projection":
    section("2-D Vector Space Projection", "TruncatedSVD of L2-normalized sparse document vectors; for qualitative visual exploration, NOT retrieval scoring.")
    st.info("The point positions are an approximation: **Top 10 rankings are computed with original high-dimensional cosine similarity**, not distance on this 2-D picture. Two close points on the chart are not necessarily close in the original vector space.")
    with st.form("projection_form"):
        two_d_query = st.text_input("Query for projection", value=st.session_state.get("last_query", "semaglutide weight loss"))
        two_d_method = st.selectbox("Weighting method (2-D)", WEIGHTINGS, index=3)
        two_d_submit = st.form_submit_button("Show 2-D projection", type="primary")
    if (two_d_submit or "projection_selection" not in st.session_state
            or st.session_state.get("projection_condition") != condition
            or st.session_state.get("projection_corpus_hash") != index.corpus_sha256):
        st.session_state["projection_selection"] = (two_d_query, two_d_method)
        st.session_state["projection_condition"] = condition
        st.session_state["projection_corpus_hash"] = index.corpus_sha256
    selected_query, selected_method = st.session_state["projection_selection"]
    projection = cached_projection(condition, index.corpus_sha256, selected_method)
    projected_query, oov = project_query(index, projection, selected_query)
    top10 = index.search(selected_query, selected_method, "Cosine", 10)
    if oov:
        st.warning("Out of vocabulary: " + ", ".join(oov))
    if projected_query is None:
        st.warning("The query has no nonzero-weight terms, so no query point can be projected.")
    qvec, _, _ = query_vector(index, selected_query, selected_method)
    cosine_scores, _ = score_documents(index, qvec, selected_method, "Cosine")
    top_positions = [hit["doc_index"] for hit in top10["hits"]]
    top_set = set(top_positions)
    remaining = [i for i in range(len(index.docs)) if i not in top_set]
    xy = projection.coordinates

    def trace_metadata(positions):
        return [[index.docs[i]["pmid"], index.docs[i]["title"][:130], f"{cosine_scores[i]:.6f}"] for i in positions]

    hover = "PMID: %{customdata[0]}<br>%{customdata[1]}<br>Original cosine: %{customdata[2]}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xy[remaining, 0], y=xy[remaining, 1], mode="markers",
        name=f"Other documents ({len(remaining)})",
        marker=dict(size=6, opacity=0.30, color="#74958a"),
        customdata=trace_metadata(remaining), hovertemplate=hover))
    if top_positions:
        fig.add_trace(go.Scatter(
            x=xy[top_positions, 0], y=xy[top_positions, 1], mode="markers+text",
            name="Original cosine Top 10",
            marker=dict(size=12, color="#e58f40", symbol="circle", line=dict(width=1, color="#6e4a28")),
            text=[str(hit["rank"]) for hit in top10["hits"]],
            textposition="top center", textfont=dict(size=10),
            customdata=trace_metadata(top_positions), hovertemplate=hover))
    if projected_query is not None:
        fig.add_trace(go.Scatter(
            x=[float(projected_query[0])], y=[float(projected_query[1])],
            name="Query", mode="markers", marker=dict(size=19, color="#d34c45", symbol="star",
                                                    line=dict(width=1, color="#77241f")),
            customdata=[[html.escape(selected_query)]],
            hovertemplate="Query: %{customdata[0]}<extra></extra>"))
    fig.update_layout(height=610, xaxis_title="SVD component 1", yaxis_title="SVD component 2",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                      hovermode="closest", margin=dict(l=35, r=25, t=65, b=40))
    st.plotly_chart(fig, width="stretch")
    variance = float(projection.explained_variance_ratio.sum())
    st.caption(f"Corpus: {len(index.docs):,} documents · Preprocessing: {condition} · Weighting: {selected_method} · "
               f"Variance explained by both components: {variance:.2%} · SVD random_state = 42 · "
               "Plot uses L2-normalized document vectors before reduction.")
    st.markdown("**Original high-dimensional Cosine Top 10**")
    st.dataframe(pd.DataFrame([{
        "Rank": h["rank"], "PMID": h["pmid"], "Title": h["title"], "Cosine": h["score"],
        "2D X": float(xy[h["doc_index"], 0]), "2D Y": float(xy[h["doc_index"], 1])
    } for h in top10["hits"]]), hide_index=True, width="stretch")
    export_rows = [{"PMID": d["pmid"], "Title": d["title"], "X": float(xy[i, 0]),
                    "Y": float(xy[i, 1]), "Original cosine": float(cosine_scores[i]),
                    "Top10 rank": next((h["rank"] for h in top10["hits"] if h["doc_index"] == i), "")}
                   for i, d in enumerate(index.docs)]
    st.download_button("Export 2-D coordinates CSV", csv_bytes(export_rows),
                       "project3_vector_projection.csv", "text/csv")

elif page == "06 Word2Vec Comparison":
    section("Word2Vec Comparison · optional learning demo",
            "CBOW / Skip-gram learned word similarity; separate from the five lexical weighting methods.")
    st.info("Word2Vec cosine similarity and TF-IDF term weight have different meanings and scales. "
            "Word2Vec does not change Project 3 VSM ranking or evaluation.")
    st.markdown("#### Project 3 Word2Vec · trained from 3,000 Main Abstracts")
    st.caption("**Additional learning demo, not a required Project 3 RQ.** "
               "Word2Vec compares *word-to-word* cosine in a 100-D learned "
               "semantic space; these scores are NOT TF-IDF weights, "
               "document-search cosine, or query-expansion results.")
    model_available = model_inventory()
    if not all(model_available.values()):
        absent = [name for name, exists in model_available.items() if not exists]
        st.warning("Missing trained models: " + ", ".join(absent))
        st.code(str(EMBEDDING_DIR), language=None)
        st.caption("Train all THREE Project 3 models from the formal 3,000 abstracts: "
                   "python src/e3_02_train_word2vec_3000.py")
    elif not EMBEDDING_REPORT_PATH.is_file():
        st.warning("Missing 3,000-document embedding_report.json.gz; run: python src/e3_02_train_word2vec_3000.py")
    else:
        try:
            _, source_header = read_corpus()
            report = read_training_report(expected_corpus_sha256=source_header["corpus_sha256"])
            st.success("Word2Vec trained on the same formal 3,000 Main Abstracts as the lexical retrieval index")
            st.caption("Word2Vec trains on sentence-split tokens using preprocessing C, min_count=2; "
                       "this page's lexical index is preprocessing " + condition + ". "
                       "When A/B/D is selected, tokenization differs, but the source documents remain the same.")
            model_label = st.selectbox("Word2Vec model (trained, read-only)",
                list(EMBEDDING_MODELS), key="w2v_model")
            word = st.text_input("Word2Vec term (one token)",
                value="semaglutide", key="w2v_term")
            other = st.text_input("Compare with (optional single token)",
                value="liraglutide", key="w2v_other")
            top_k = st.slider("Top-K similar words", 1, 20, 5, key="w2v_topk")
            model_path = EMBEDDING_DIR / EMBEDDING_MODELS[model_label]
            w2v = cached_word2vec(model_label, str(model_path),
                                 model_path.stat().st_size, model_path.stat().st_mtime_ns)
            experiment = report["experiments"].get(EMBEDDING_MODELS[model_label].removesuffix(".model"), {})
            params = experiment.get("params", {})
            st.caption(f"{model_label} · Vocab {len(w2v.wv):,} · "
                       f"Vector dimension {w2v.wv.vector_size} · "
                       f"Window {params.get('window', 'unknown')} · "
                       f"Epochs {params.get('epochs', 'unknown')}")
            outcome = compare_embeddings(w2v, word, other, top_k)
            if not outcome["term_exists"]:
                st.warning(f"OOV: '{outcome['term']}' is not in this Word2Vec model.")
            if other and not outcome["other_exists"]:
                st.warning(f"OOV comparison word: '{outcome['other']}'")
            for unknown, suggested in outcome["suggestions"].items():
                if suggested:
                    st.caption(f"Similar known spellings for '{unknown}': {', '.join(suggested)}")
            if outcome["pair_cosine"] is not None:
                st.metric("Word2Vec pair cosine", f"{outcome['pair_cosine']:.4f}")
                st.caption("Cosine of learned 100-D WORD vectors; not a document score.")
            if outcome["neighbors"]:
                st.markdown("**Actual nearest neighbors from trained model**")
                st.dataframe(pd.DataFrame(outcome["neighbors"]),
                             hide_index=True, width="stretch")
                st.download_button("Export Word2Vec neighbors CSV",
                    csv_bytes(outcome["neighbors"]), "project3_w2v_neighbors.csv", "text/csv")
            st.caption("Models trained offline from Project 3 Corpus 3000. Word2Vec is not used for Project 3 retrieval "
                       "or relevance evaluation. Semantic neighbors may not be synonyms.")
        except (FileNotFoundError, ValueError, ImportError, OSError) as exc:
            st.warning(f"Word2Vec unavailable: {type(exc).__name__}: {exc}")
            st.caption("Try installing gensim with: python -m pip install -r requirements.txt")

st.divider()
st.caption("NCKU Project #3 | All retrieval, weighting, evaluation and Word2Vec use formal 3,000 JATS Main Abstracts | No live network data")
