"""Single-file HTML rendering of a report.

The output is one self-contained document: no external stylesheet, no script,
no network request. A report that phones home when opened would be a strange
thing for a library about surveillance to produce, and a self-contained file is
also the only kind that still works when it is emailed to a supervisor.

The layout puts the caveats above the numbers on purpose. Everything this
library measures is easy to quote out of context, and a limitation printed
after the table is a limitation nobody reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from synthwatch.report.report import Report

__all__ = ["to_html"]

LARGE_VALUE = 1000.0
SMALL_VALUE = 0.01
"""Outside this range a value is rendered in significant figures, not decimals."""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ report.title }}</title>
<style>
  :root { color-scheme: light dark; --fg:#1a1a1a; --muted:#5b5b5b; --bg:#fbfbfa;
          --card:#ffffff; --line:#e2e0dc; --warn:#8a5a00; --warnbg:#fff8e8; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e6e3; --muted:#a3a09b; --bg:#14140f; --card:#1d1d18;
            --line:#333029; --warn:#e8bc70; --warnbg:#2a2113; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
         font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }
  main { max-width: 60rem; margin: 0 auto; }
  h1 { font-size:1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }
  h2 { font-size:1.1rem; margin:2.5rem 0 .75rem; letter-spacing:-.01em; }
  h3 { font-size:.95rem; margin:0 0 .5rem; }
  .sub { color:var(--muted); margin:0 0 2rem; font-size:.9rem; }
  .caveats { background:var(--warnbg); border:1px solid var(--line);
             border-left:3px solid var(--warn); border-radius:6px; padding:1rem 1.25rem; }
  .caveats h2 { margin:0 0 .5rem; font-size:.95rem; color:var(--warn); }
  .caveats ul { margin:0; padding-left:1.1rem; }
  .caveats li { margin:.4rem 0; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:.75rem; }
  .stat { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:.85rem 1rem; }
  .stat .k { color:var(--muted); font-size:.78rem; text-transform:uppercase;
             letter-spacing:.04em; display:block; }
  .stat .v { font-size:1.35rem; font-variant-numeric:tabular-nums; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:6px;
          padding:1rem 1.25rem; margin-bottom:1rem; }
  .card .caveat { color:var(--muted); font-size:.85rem; border-top:1px solid var(--line);
                  margin-top:.85rem; padding-top:.75rem; }
  .members { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8rem;
             color:var(--muted); word-break:break-all; }
  .scroll { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:.85rem; }
  th, td { text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { font-weight:600; color:var(--muted); font-size:.78rem; text-transform:uppercase;
       letter-spacing:.04em; white-space:nowrap; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .lim { color:var(--muted); font-size:.85rem; }
  .empty { color:var(--muted); font-style:italic; }
  footer { color:var(--muted); font-size:.8rem; margin-top:3rem; border-top:1px solid var(--line);
           padding-top:1rem; }
  code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em; }
</style>
</head>
<body>
<main>
  <h1>{{ report.title }}</h1>
  <p class="sub">
    {{ corpus.n_posts }} posts from {{ corpus.n_accounts }} accounts
    {%- if corpus.first_post %}, {{ corpus.first_post[:10] }} to {{ corpus.last_post[:10] }}{% endif %}.
    Source: <code>{{ corpus.source or "unspecified" }}</code>.
    Generated {{ report.generated_at.strftime("%Y-%m-%d %H:%M") }} UTC
    by synthwatch {{ report.library_version }}.
  </p>

  <section class="caveats">
    <h2>How to read this</h2>
    <ul>{% for caveat in report.caveats %}<li>{{ caveat }}</li>{% endfor %}</ul>
  </section>

  <h2>Corpus</h2>
  <div class="grid">
    <div class="stat"><span class="k">Posts</span><span class="v">{{ corpus.n_posts }}</span></div>
    <div class="stat"><span class="k">Accounts</span><span class="v">{{ corpus.n_accounts }}</span></div>
    <div class="stat"><span class="k">Platforms</span><span class="v">{{ corpus.platforms|join(", ") or "—" }}</span></div>
    {% if ingest %}
    <div class="stat"><span class="k">Rows skipped at load</span><span class="v">{{ ingest.n_skipped }}</span></div>
    {% endif %}
  </div>
  {% if ingest and ingest.warnings %}
  <ul class="lim">{% for warning in ingest.warnings %}<li>{{ warning }}</li>{% endfor %}</ul>
  {% endif %}

  <h2>Coordination</h2>
  {% if coordination %}
  <div class="grid">
    <div class="stat"><span class="k">Clusters</span><span class="v">{{ report.cards|length }}</span></div>
    <div class="stat"><span class="k">Accounts in clusters</span><span class="v">{{ accounts_in_clusters }}</span></div>
    <div class="stat"><span class="k">Edges</span><span class="v">{{ coordination.stats.n_edges }}</span></div>
    <div class="stat"><span class="k">Window</span><span class="v">{{ (coordination.config.window_seconds / 60)|round(1) }} min</span></div>
    {% if coordination.null_model %}
    <div class="stat"><span class="k">Null model p</span><span class="v">{{ coordination.null_model.p_value }}</span></div>
    {% endif %}
  </div>
  <p class="lim">{{ coordination.caveat }}</p>
  {% else %}
  <p class="empty">No coordination analysis was run.</p>
  {% endif %}

  {% if report.cards %}
  <h2>Clusters</h2>
  {% for card in report.cards %}
  <article class="card">
    <h3>{{ card.cluster_id }} — {{ card.size }} accounts</h3>
    <div class="grid">
      <div class="stat"><span class="k">Density</span><span class="v">{{ card.density|round(2) }}</span></div>
      <div class="stat"><span class="k">Mean similarity</span><span class="v">{{ card.mean_similarity|round(3) }}</span></div>
      <div class="stat"><span class="k">Median lag</span><span class="v">{{ card.median_lag_seconds|round(0)|int }}s</span></div>
      <div class="stat"><span class="k">Co-posts</span><span class="v">{{ card.total_pairs }}</span></div>
      {% if card.median_account_age_days is not none %}
      <div class="stat"><span class="k">Median age</span><span class="v">{{ card.median_account_age_days|round(0)|int }}d</span></div>
      {% endif %}
      {% if card.quiet_window_free_members is not none %}
      <div class="stat"><span class="k">Members with no quiet hours</span><span class="v">{{ card.quiet_window_free_members }}/{{ card.size }}</span></div>
      {% endif %}
    </div>
    <p class="members">{{ card.members|join(" · ") }}</p>
    {% if card.example_post_pairs %}
    <p class="lim">Example co-posts:
      {% for pair in card.example_post_pairs %}<code>{{ pair[0] }} ↔ {{ pair[1] }}</code>{% if not loop.last %}, {% endif %}{% endfor %}
    </p>
    {% endif %}
    <p class="caveat">{{ card.caveat }}</p>
  </article>
  {% endfor %}
  {% endif %}

  <h2>Feature distributions</h2>
  <p class="lim">Across the whole corpus. Coverage is the share of accounts the
  feature could be measured for; a low coverage makes the quantiles beside it
  describe a subset, not the corpus.</p>
  <div class="scroll">
  <table>
    <thead><tr>
      <th>Feature</th><th class="num">Coverage</th><th class="num">p10</th>
      <th class="num">median</th><th class="num">p90</th><th>Known limitation</th>
    </tr></thead>
    <tbody>
    {% for feature in report.features %}
      <tr>
        <td><code>{{ feature.spec.name }}</code><br><span class="lim">{{ feature.spec.description }}</span></td>
        <td class="num">{{ (feature.coverage * 100)|round(0)|int }}%</td>
        <td class="num">{{ fmt(feature.p10) }}</td>
        <td class="num">{{ fmt(feature.median) }}</td>
        <td class="num">{{ fmt(feature.p90) }}</td>
        <td class="lim">{{ feature.spec.limitation }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>

  <h2>Parameters</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>Setting</th><th>Value</th></tr></thead>
    <tbody>
    {% for group, settings in report.configs.items() %}
      {% if settings is mapping %}
        {% for key, value in settings.items() %}
        <tr><td><code>{{ group }}.{{ key }}</code></td><td><code>{{ value }}</code></td></tr>
        {% endfor %}
      {% else %}
        <tr><td><code>{{ group }}</code></td><td><code>{{ settings }}</code></td></tr>
      {% endif %}
    {% endfor %}
    </tbody>
  </table>
  </div>

  <footer>
    Produced by <a href="https://github.com/Locotomate21/Synth_Watch">synthwatch</a>
    {{ report.library_version }}. This document reports aggregate and cluster-level
    behaviour. It contains no verdict about any individual account, and it should
    not be used to produce one.
  </footer>
</main>
</body>
</html>
"""


def _format_number(value: float | None) -> str:
    """Render a number for a table cell, distinguishing zero from unknown."""
    if value is None:
        return "—"
    if value == 0:
        return "0"
    if abs(value) >= LARGE_VALUE or abs(value) < SMALL_VALUE:
        return f"{value:.3g}"
    return f"{value:.2f}"


def to_html(report: Report, path: Path | None = None) -> str:
    """Render ``report`` as a self-contained HTML document.

    Args:
        report: The report to render.
        path: Written to when given.

    Raises:
        ImportError: If Jinja2 is not installed. It lives in the ``report``
            extra so that the analysis core stays dependency-light.
    """
    try:
        # Deliberately late: Jinja2 lives in the `report` extra so that loading
        # the analysis core never requires it.
        from jinja2 import Environment, select_autoescape  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - exercised by the extra being absent
        msg = 'HTML export needs Jinja2. Install the report extra: pip install "synthwatch[report]"'
        raise ImportError(msg) from error

    environment = Environment(autoescape=select_autoescape(default_for_string=True))
    environment.globals["fmt"] = _format_number
    payload = report.as_dict()
    rendered = environment.from_string(TEMPLATE).render(
        report=report,
        corpus=payload["corpus"],
        ingest=payload["ingest"],
        coordination=payload["coordination"],
        accounts_in_clusters=sum(card.size for card in report.cards),
    )
    if path is not None:
        path.write_text(rendered, encoding="utf-8")
    return rendered
