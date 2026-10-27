"""Presentation-only Plotly charts for Project 3.

Separates figure construction from the Streamlit page controller. The charts
receive evaluated metrics as input and **never** recalculate retrieval scores,
ranking or relevance judgments.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import plotly.graph_objects as go

from e4_03_term_weighting import WEIGHTINGS


EVALUATION_METRICS = (
    "Precision@10", "Recall@10", "MRR@10", "nDCG@10",
)


def build_grouped_evaluation_chart(means: Sequence[Mapping[str, object]]) -> go.Figure:
    """Plot independent IR metrics as grouped (not stacked) bars.

    X axis contains the four metrics. Each metric has one bar per weighting
    method; height is the mean metric value computed by E7 (range 0..1).
    Missing or malformed results raise an error instead of silently plotting
    misleading values. This is deliberately a pure presentation function.
    """
    lookup = {str(record["Weighting"]): record for record in means}
    if len(means) != len(WEIGHTINGS) or set(lookup) != set(WEIGHTINGS):
        raise ValueError("Grouped chart requires exactly one result for each weighting method")

    fig = go.Figure()
    for method in WEIGHTINGS:
        record = lookup[method]
        values = []
        for metric in EVALUATION_METRICS:
            value = record.get(metric)
            if value is None or not isinstance(value, (int, float)):
                raise ValueError(f"Missing or non-numeric {metric} for {method}")
            value = float(value)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"Out-of-range {metric} for {method}: {value}")
            values.append(value)
        fig.add_trace(go.Bar(
            name=method,
            x=list(EVALUATION_METRICS),
            y=values,
            # Labels are purely presentational; values are the unchanged E7 means.
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
            textangle=-55,
            textfont=dict(size=15, color="#233C37", family="Arial, sans-serif"),
            cliponaxis=False,
            hovertemplate=(
                f"Method: {method}<br>Metric: %{{x}}<br>Mean score: %{{y:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Average Evaluation Scores across 10 Queries",
                   x=0.5, xanchor="center", y=0.99, yanchor="top",
                   font=dict(size=23, color="#183E35", family="Arial, sans-serif")),
        font=dict(size=16, color="#243A35", family="Arial, sans-serif"),
        barmode="group",
        bargap=0.18,
        bargroupgap=0.04,
        xaxis=dict(title=None, categoryorder="array",
                   categoryarray=list(EVALUATION_METRICS),
                   tickfont=dict(size=18, color="#183E35",
                                 family="Arial Black, Arial, sans-serif"),
                   automargin=True),
        yaxis=dict(title=dict(text="Mean Score",
                              font=dict(size=20, color="#183E35",
                                        family="Arial, sans-serif")),
                   range=[0, 1], tick0=0, dtick=0.2, tickformat=".1f",
                   tickfont=dict(size=17, color="#183E35"),
                   automargin=True, zeroline=False),
        legend=dict(title=None, orientation="h", y=1.15, x=0.5,
                    xanchor="center", yanchor="bottom",
                    font=dict(size=18, color="#183E35"),
                    bgcolor="rgba(255,255,255,0)"),
        margin=dict(l=95, r=35, t=150, b=95),
        height=660,
        uniformtext=dict(mode="show", minsize=13),
    )
    return fig


def build_two_term_weight_chart(rows: Sequence[Mapping[str, object]]) -> go.Figure:
    """Side-by-side document TF-IDF weights for the two selected terms."""
    if len(rows) != 2 or len({str(row["Term"]) for row in rows}) != 2:
        raise ValueError("Exactly two different term rows required")
    fig = go.Figure()
    for method in ("Raw TF-IDF", "Log TF-IDF"):
        values = [float(row[method]) for row in rows]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid TF-IDF weight")
        fig.add_trace(go.Bar(
            name=method, x=[str(row["Term"]) for row in rows], y=values,
            hovertemplate=f"Term: %{{x}}<br>{method}: %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", height=340,
        yaxis_title="Weight in selected document (not cosine)",
        xaxis_title="Normalized term",
        legend=dict(orientation="h", y=1.16, x=0),
        margin=dict(l=40, r=15, t=55, b=45))
    return fig


# Weighting comparison charts: one shared numeric axis in each figure.
# Intentionally leave E4 weighting formulas and values entirely untouched.
def _five_term_rows(weight_rows):
    """Validate five ordered method rows for one normalized term."""
    if (len(weight_rows) != len(WEIGHTINGS)
            or [row["Weighting"] for row in weight_rows] != list(WEIGHTINGS)
            or len({row["Term"] for row in weight_rows}) != 1):
        raise ValueError("Exactly five ordered rows for one term are required")
    for row in weight_rows:
        for key in ("Query Weight", "Document Weight"):
            val = float(row[key])
            if not math.isfinite(val) or val < 0:
                raise ValueError("Weight must be finite and nonnegative")
    return str(weight_rows[0]["Term"])


def _weight_text(value):
    """Prevent nonzero tiny values from being rendered as misleading zero."""
    value = float(value)
    if value == 0:
        return "0"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _weight_range(values):
    peak = max(values, default=0.0)
    return [0.0, peak * 1.34 if peak > 0 else 1.0]


def _style_weight_bar_for_projection(fig, grouped=False):
    """02/A3–A5 presentation only; original bar values are untouched."""
    fig.update_traces(textfont=dict(size=17, family="Arial, sans-serif"),
                      textposition="outside", cliponaxis=False)
    fig.update_layout(
        font=dict(size=16, family="Arial, sans-serif", color="#243B38"),
        title=dict(font=dict(size=22, family="Arial, sans-serif")),
        xaxis=dict(tickfont=dict(size=15), title_font=dict(size=17)),
        yaxis=dict(title=None, tickfont=dict(size=17, family="Arial Black, Arial, sans-serif"),
                   automargin=True),
        margin=dict(l=160, r=118, t=85, b=120 if grouped else 74),
        height=590 if grouped else 485,
    )
    if grouped:
        fig.update_layout(legend=dict(orientation="h", y=-0.19,
            x=0, xanchor="left", yanchor="top", font=dict(size=17)))
    return fig


def build_single_term_five_weight_chart(weight_rows):
    """A1: one horizontal bar chart, common X axis for five document weights."""
    term = _five_term_rows(weight_rows)
    values = [float(row["Document Weight"]) for row in weight_rows]
    fig = go.Figure(go.Bar(
        name="Document weight", orientation="h",
        y=list(WEIGHTINGS), x=values,
        text=[_weight_text(v) for v in values], textposition="outside",
        cliponaxis=False, marker_color="#287569",
        customdata=[term] * len(values),
        hovertemplate="Term: %{customdata}<br>Method: %{y}<br>Document weight: %{x:.8f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{term} · five document weighting methods",
        height=400, showlegend=False, bargap=0.30,
        xaxis=dict(title="Document term weight (shared scale)", range=_weight_range(values), zeroline=True),
        yaxis=dict(title="Weighting method", categoryorder="array",
                   categoryarray=list(reversed(WEIGHTINGS))),
        margin=dict(l=115, r=60, t=60, b=55),
    )
    return _style_weight_bar_for_projection(fig, grouped=False)


def build_query_document_comparison_chart(weight_rows):
    """A2: five grouped horizontal Query vs Document pairs with common X scale."""
    term = _five_term_rows(weight_rows)
    q_values = [float(row["Query Weight"]) for row in weight_rows]
    d_values = [float(row["Document Weight"]) for row in weight_rows]
    fig = go.Figure()
    for label, vals, color in (
        ("Query weight", q_values, "#D88B31"),
        ("Document weight", d_values, "#287569"),
    ):
        fig.add_trace(go.Bar(
            name=label, orientation="h", y=list(WEIGHTINGS), x=vals,
            text=[_weight_text(v) for v in vals], textposition="outside",
            cliponaxis=False, marker_color=color,
            customdata=[term] * len(vals),
            hovertemplate=f"Term: %{{customdata}}<br>Method: %{{y}}<br>{label}: %{{x:.8f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"{term} · Query vs Document term weight", x=0.0, xanchor="left", y=0.985),
        barmode="group", bargap=0.26, bargroupgap=0.05, height=500,
        xaxis=dict(title="Term weight (shared scale)",
                   range=_weight_range(q_values + d_values), zeroline=True),
        yaxis=dict(title="Weighting method", categoryorder="array",
                   categoryarray=list(reversed(WEIGHTINGS))),
        # Put the legend BELOW the plot; it cannot overlap the title.
        legend=dict(orientation="h", y=-0.26, x=0.0, yanchor="top", xanchor="left"),
        margin=dict(l=120, r=70, t=82, b=110),
    )
    return _style_weight_bar_for_projection(fig, grouped=True)


def build_two_term_five_weight_chart(weight_rows):
    """A3: two document terms × five methods on one grouped HORIZONTAL figure."""
    terms = list(dict.fromkeys(row["Term"] for row in weight_rows))
    if len(terms) != 2 or len(weight_rows) != 2 * len(WEIGHTINGS):
        raise ValueError("Exactly two terms and five methods required")
    lookup = {}
    for row in weight_rows:
        key = (row["Term"], row["Weighting"])
        if key in lookup:
            raise ValueError("Duplicate term / method in A3")
        value = float(row["Document Weight"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid weight")
        lookup[key] = value
    if set(lookup) != {(term, method) for term in terms for method in WEIGHTINGS}:
        raise ValueError("Missing term / method in A3")
    fig = go.Figure()
    all_values = []
    for term, color in zip(terms, ("#287569", "#D88B31")):
        vals = [lookup[(term, method)] for method in WEIGHTINGS]
        all_values.extend(vals)
        fig.add_trace(go.Bar(
            name=term, orientation="h", y=list(WEIGHTINGS), x=vals,
            text=[_weight_text(v) for v in vals], textposition="outside",
            cliponaxis=False, marker_color=color,
            hovertemplate=f"Term: {term}<br>Method: %{{y}}<br>Document weight: %{{x:.8f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"{terms[0]} vs {terms[1]} · five document weights",
                   x=0, xanchor="left", y=0.985),
        barmode="group", bargap=0.24, bargroupgap=0.05,
        height=500,
        xaxis=dict(title="Document term weight (shared scale)",
                   range=_weight_range(all_values), zeroline=True),
        yaxis=dict(title="Weighting method", categoryorder="array",
                   categoryarray=list(reversed(WEIGHTINGS))),
        legend=dict(orientation="h", y=-0.26, x=0.0, yanchor="top", xanchor="left"),
        margin=dict(l=120, r=70, t=82, b=110),
    )
    return _style_weight_bar_for_projection(fig, grouped=True)


def build_shared_umap_ranking_chart(index, projection, method, result,
                                    labels, query_xy=None, axis_ranges=None):
    """Overlay ONE method's unchanged M08 Top 10 on the shared Log TF-IDF UMAP.

    The five panels are NOT five independently fitted UMAPs. They have matching
    document/query coordinates; method-specific marker membership is the only
    difference. Plotting does not recompute cosine or reorder any hits.
    """
    import html
    import numpy as np
    from e8_02_shared_umap import plot_inputs

    content = plot_inputs(index, projection, method, result, labels, query_xy)
    xy = projection.coords
    background = content["other_indices"]
    top = content["top"]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name="Other documents", mode="markers", showlegend=True,
        x=xy[background, 0], y=xy[background, 1],
        marker=dict(size=4, color="#C7D1CE", opacity=0.34),
        hoverinfo="skip",
    ))
    if top:
        indices = [hit["doc_index"] for hit in top]
        hover = [f"{html.escape(hit['marker'])} · Rank {hit['rank']}"
                 f"<br>PMID {html.escape(hit['pmid'])}"
                 f"<br>{html.escape(hit['title'])}"
                 f"<br>Original Cosine {hit['score']:.6f}" for hit in top]
        fig.add_trace(go.Scattergl(
            name="Top 10 (ranks 1–3 labeled)", mode="markers+text",
            x=xy[indices, 0], y=xy[indices, 1],
            text=[str(hit["rank"]) if hit["rank"] <= 3 else "" for hit in top],
            textposition="top center", textfont=dict(size=12, color="#985214"),
            marker=dict(size=10.5, color="#E88929", opacity=0.95,
                        line=dict(width=1.2, color="#A85E22")),
            hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        ))
    if query_xy is not None:
        fig.add_trace(go.Scatter(
            name="Query (shared reference)", mode="markers",
            x=[query_xy[0]], y=[query_xy[1]],
            marker=dict(symbol="star", size=17, color="#C83B3B",
                        line=dict(color="#8E2828", width=1.4)),
            hovertemplate="Query · reference UMAP projection<extra></extra>",
        ))
    if axis_ranges is None:
        finite_xy = xy[np.isfinite(xy).all(axis=1)]
        if query_xy is not None:
            finite_xy = np.vstack((finite_xy, query_xy))
        axis_ranges = shared_umap_axis_ranges(finite_xy)
    fig.update_layout(
        title=dict(text=method, x=0.02, xanchor="left", font=dict(size=17)),
        height=405, autosize=True, template="plotly_white",
        margin=dict(l=45, r=16, t=60, b=92),
        legend=dict(orientation="h", y=-0.21, x=0.0, yanchor="top",
                    font=dict(size=10), itemwidth=30),
        xaxis=dict(title="UMAP 1", range=axis_ranges[0], zeroline=False,
                   showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(title="UMAP 2", range=axis_ranges[1], zeroline=False,
                   showgrid=False, scaleanchor="x", scaleratio=1,
                   tickfont=dict(size=9)),
        hovermode="closest", showlegend=True,
        uirevision="project3-shared-umap-v1",
    )
    return fig


def shared_umap_axis_ranges(coords):
    """Compute identical, padded coordinate extents for all five charts."""
    import numpy as np
    coords = np.asarray(coords, dtype=float)
    finite = coords[np.isfinite(coords).all(axis=1)]
    if not len(finite):
        raise ValueError("No finite UMAP points to plot")
    ranges = []
    for axis in range(2):
        low, high = float(finite[:, axis].min()), float(finite[:, axis].max())
        pad = max((high - low) * 0.06, 0.5)
        ranges.append([low - pad, high + pad])
    return ranges
