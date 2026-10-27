"""UI chart test: metric values are preserved; grouping is not stacking."""
import ast
import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e4_03_term_weighting import WEIGHTINGS
from project3_charts import EVALUATION_METRICS, build_grouped_evaluation_chart


def _read_means():
    with (ROOT / "results" / "mean_metrics.csv").open(encoding="utf-8-sig", newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def test_grouped_evaluation_chart_matches_frozen_csv():
    means = _read_means()
    numeric_means = [
        {k: float(v) if k in EVALUATION_METRICS else v for k, v in row.items()}
        for row in means
    ]
    fig = build_grouped_evaluation_chart(numeric_means)
    assert fig.layout.barmode == "group"
    assert list(fig.layout.yaxis.range) == [0, 1]
    assert tuple(trace.name for trace in fig.data) == WEIGHTINGS
    assert len(fig.data) == 5
    for method, trace in zip(WEIGHTINGS, fig.data):
        assert tuple(trace.x) == EVALUATION_METRICS
        record = next(row for row in numeric_means if row["Weighting"] == method)
        for label, value in zip(trace.x, trace.y):
            assert value == pytest.approx(record[label], abs=1e-13)
    assert fig.data[-1].y[-1] == pytest.approx(0.9394434793486648)


def test_grouped_evaluation_chart_rejects_wrong_records():
    with pytest.raises(ValueError):
        build_grouped_evaluation_chart([])
    rows = [
        {"Weighting": name, **{metric: 0.5 for metric in EVALUATION_METRICS}}
        for name in WEIGHTINGS
    ]
    rows[0]["Precision@10"] = 1.2
    with pytest.raises(ValueError, match="Out-of-range"):
        build_grouped_evaluation_chart(rows)


def test_m09_calls_chart_without_altering_e7():
    app = (ROOT / "src/m09_app.py").read_text(encoding="utf-8")
    ast.parse(app)
    assert "st.bar_chart(" not in app
    assert "build_grouped_evaluation_chart(means)" in app
    assert "st.dataframe(pd.DataFrame(means)" in app
    assert "Export mean CSV" in app
