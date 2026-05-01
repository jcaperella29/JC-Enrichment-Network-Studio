import math
from typing import Dict, Tuple
from typing import Dict
import networkx as nx
import numpy as np
import base64
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import io
import json
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from io import StringIO

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update


# ----------------------------
# Helpers
# ----------------------------
REQUIRED_RAW_COLS_NOTE = (
    "Raw CSV should be long-format: one row per membership edge (gene ↔ pathway). "
    "You’ll map the item + group columns after upload."
)

import plotly.express as px
import math


def edge_weight_from_adj_p(p: float) -> float:
    """Convert adjusted p-value to a 'bigger = more significant' weight."""
    if p is None:
        return 0.0
    try:
        p = float(p)
    except Exception:
        return 0.0
    # clamp to avoid log(0)
    p = max(p, 1e-300)
    return max(0.0, -math.log10(p))


def minmax_scale(x: float, xmin: float, xmax: float, out_min: float, out_max: float) -> float:
    """Map x from [xmin,xmax] to [out_min,out_max]."""
    if xmax == xmin:
        return (out_min + out_max) / 2.0
    return out_min + (x - xmin) * (out_max - out_min) / (xmax - xmin)


def term_color_map(terms):
    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Dark24
    return {t: palette[i % len(palette)] for i, t in enumerate(sorted(terms))}


def parse_upload(contents: str) -> pd.DataFrame:
    """Parse Dash upload contents into a pandas DataFrame (CSV)."""
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

DEMO_CSV_PATH = Path(__file__).parent / "TEST.csv"


def load_demo_dataframe() -> pd.DataFrame:
    """
    Load the bundled demo file and convert Enrichr-style term -> semicolon genes
    into long-format gene -> term rows for the network builder.
    """
    df = pd.read_csv(DEMO_CSV_PATH)

    required = {"term", "genes", "adjusted_pvalue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Demo CSV is missing required columns: {sorted(missing)}")

    long_df = (
        df.assign(gene=df["genes"].astype(str).str.split(";"))
          .explode("gene")
          .assign(gene=lambda x: x["gene"].astype(str).str.strip())
    )

    long_df = long_df[long_df["gene"].ne("")]
    long_df = long_df[["gene", "term", "adjusted_pvalue"]].drop_duplicates()

    return long_df


COLUMN_MAPPING_PRESETS = [
    {"label": "Custom long-format CSV", "value": "custom_long"},
    {"label": "Enrichr-style: term + semicolon genes", "value": "enrichr"},
    {"label": "g:Profiler / gprofiler2-style", "value": "gprofiler"},
    {"label": "clusterProfiler-style", "value": "clusterprofiler"},
    {"label": "GSEA / MSigDB-style", "value": "gsea_msigdb"},
]


def _normalize_col_name(name: str) -> str:
    """Normalize column names for forgiving preset matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a dataframe column from likely names, ignoring case/punctuation."""
    normalized = {_normalize_col_name(c): c for c in df.columns}
    for candidate in candidates:
        hit = normalized.get(_normalize_col_name(candidate))
        if hit is not None:
            return hit
    return None


def _guess_long_format_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Guess item/group/weight columns for already-long enrichment edge tables."""
    item_col = _find_col(
        df,
        [
            "gene", "genes", "item", "items", "symbol", "gene_symbol", "geneid",
            "gene_id", "target", "feature", "node", "protein",
        ],
    )
    group_col = _find_col(
        df,
        [
            "term", "pathway", "pathways", "group", "description", "name",
            "term_name", "termid", "term_id", "geneset", "gene_set",
        ],
    )
    weight_col = _find_col(
        df,
        [
            "adjusted_pvalue", "adjusted_p_value", "adjusted p-value", "adjusted p value",
            "adjusted.p.value", "padj", "p.adjust", "p_adjust", "qvalue", "q_value",
            "fdr", "fdr q-val", "fdr_q_val", "p_value", "pvalue", "p.val", "pval",
        ],
    )
    return item_col, group_col, weight_col


def _split_gene_memberships(value) -> list[str]:
    """Split common enrichment gene-list cells into individual gene symbols/items."""
    import re

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    # clusterProfiler often uses '/', Enrichr often ';', many CSV exports use ','.
    # Also handle pipes and whitespace as secondary separators.
    parts = re.split(r"[;,/|]+", text)
    genes = [p.strip() for p in parts if p and p.strip()]
    return genes


def _expand_membership_table(
    df: pd.DataFrame,
    term_col: str,
    genes_col: str,
    weight_col: str | None,
    preset_label: str,
) -> tuple[pd.DataFrame, str, str, str | None, str]:
    """Convert term -> gene-list tables into long gene -> term membership rows."""
    records = []

    for _, row in df.iterrows():
        term = row.get(term_col)
        genes = _split_gene_memberships(row.get(genes_col))
        if term is None or pd.isna(term) or not str(term).strip():
            continue
        for gene in genes:
            rec = {"gene": gene, "term": str(term).strip()}
            if weight_col:
                rec["adjusted_pvalue"] = row.get(weight_col)
            records.append(rec)

    if not records:
        raise ValueError(
            f"{preset_label} preset found columns, but no gene memberships could be expanded. "
            "Check the gene-list separator or choose Custom long-format CSV."
        )

    long_df = pd.DataFrame(records).drop_duplicates()
    item_col = "gene"
    group_col = "term"
    out_weight_col = "adjusted_pvalue" if "adjusted_pvalue" in long_df.columns else None

    msg = (
        f"Applied {preset_label} preset: expanded {len(long_df):,} gene-term edges "
        f"from {long_df['gene'].nunique():,} genes/items and {long_df['term'].nunique():,} pathways/terms."
    )
    return long_df, item_col, group_col, out_weight_col, msg


def apply_column_mapping_preset(df: pd.DataFrame, preset: str) -> tuple[pd.DataFrame, str | None, str | None, str | None, str]:
    """
    Apply a column mapping preset.

    For term -> gene-list formats, this converts the table into the long-format
    gene ↔ pathway edge table used by the network builder.
    """
    preset = preset or "custom_long"

    if preset == "custom_long":
        item_col, group_col, weight_col = _guess_long_format_columns(df)
        msg = "Applied Custom long-format preset."
        if item_col and group_col:
            msg += f" Guessed item={item_col}, group={group_col}"
            if weight_col:
                msg += f", weight={weight_col}"
            msg += "."
        else:
            msg += " Select item and group columns manually."
        return df, item_col, group_col, weight_col, msg

    if preset == "enrichr":
        term_col = _find_col(df, ["term", "Term", "name", "pathway", "description"])
        genes_col = _find_col(df, ["genes", "Genes", "overlapping genes", "overlap_genes"])
        weight_col = _find_col(df, ["adjusted_pvalue", "Adjusted.P.value", "Adjusted P-value", "adjusted p value", "padj", "fdr", "pvalue", "p_value"])
        if not term_col or not genes_col:
            raise ValueError("Enrichr preset needs a term column and a genes column.")
        return _expand_membership_table(df, term_col, genes_col, weight_col, "Enrichr-style")

    if preset == "gprofiler":
        term_col = _find_col(df, ["term_name", "name", "term", "native", "term_id", "description"])
        genes_col = _find_col(df, ["intersection", "intersections", "genes", "query", "members"])
        weight_col = _find_col(df, ["p_value", "pvalue", "adjusted_pvalue", "padj", "fdr", "qvalue"])
        if not term_col or not genes_col:
            raise ValueError("g:Profiler preset needs a term/name column and an intersection/genes column.")
        return _expand_membership_table(df, term_col, genes_col, weight_col, "g:Profiler/gprofiler2-style")

    if preset == "clusterprofiler":
        term_col = _find_col(df, ["Description", "description", "term", "pathway", "ID", "id"])
        genes_col = _find_col(df, ["geneID", "gene_id", "genes", "Genes", "core_enrichment"])
        weight_col = _find_col(df, ["p.adjust", "p_adjust", "padj", "qvalue", "pvalue", "p_value"])
        if not term_col or not genes_col:
            raise ValueError("clusterProfiler preset needs Description/ID and geneID/core_enrichment columns.")
        return _expand_membership_table(df, term_col, genes_col, weight_col, "clusterProfiler-style")

    if preset == "gsea_msigdb":
        term_col = _find_col(df, ["NAME", "name", "Description", "description", "term", "pathway", "geneset", "gene_set"])
        genes_col = _find_col(df, ["core_enrichment", "leading_edge", "leading edge", "genes", "Genes", "members"])
        weight_col = _find_col(df, ["FDR q-val", "FDR.q.val", "fdr", "qvalue", "q_value", "p.adjust", "padj", "pvalue", "p_value"])
        if not term_col or not genes_col:
            raise ValueError("GSEA/MSigDB preset needs a NAME/term column and core_enrichment/genes column.")
        return _expand_membership_table(df, term_col, genes_col, weight_col, "GSEA/MSigDB-style")

    raise ValueError(f"Unknown column mapping preset: {preset}")


import math
import pandas as pd
import networkx as nx


def build_bipartite_graph(
    df: pd.DataFrame,
    item_col: str,
    group_col: str,
    weight_col: str | None = None,
) -> nx.Graph:
    """
    Build an undirected bipartite graph from long-form rows.

    If weight_col is provided (e.g. adjusted p-value),
    edge weight is transformed to:
        weight = -log10(adjusted_pvalue)

    This means:
      - smaller p-values → stronger edges
      - filtering with weight >= threshold behaves correctly
    """
    g = nx.Graph()

    cols = [item_col, group_col] + ([weight_col] if weight_col else [])
    tmp = df[cols].copy()

    # Clean strings
    tmp[item_col] = tmp[item_col].astype(str).str.strip()
    tmp[group_col] = tmp[group_col].astype(str).str.strip()

    # Drop junk
    tmp = tmp.dropna(subset=[item_col, group_col])
    tmp = tmp[(tmp[item_col] != "") & (tmp[group_col] != "")]
    tmp = tmp.drop_duplicates(subset=[item_col, group_col])

    # Add nodes
    items = tmp[item_col].unique().tolist()
    groups = tmp[group_col].unique().tolist()
    for it in items:
        g.add_node(f"item::{it}", label=it, node_type="item")

    for gr in groups:
        g.add_node(f"group::{gr}", label=gr, node_type="group")

    if weight_col and weight_col in tmp.columns:
        w_raw = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(1.0)

        for it, gr, raw in zip(tmp[item_col], tmp[group_col], w_raw):
            raw = float(raw)
            w_plot = edge_weight_from_adj_p(raw)

            g.add_edge(
                f"item::{it}",
                f"group::{gr}",
                weight_raw=raw,
                weight_plot=w_plot
            )
    else:
        for it, gr in zip(tmp[item_col], tmp[group_col]):
            g.add_edge(
                f"item::{it}",
                f"group::{gr}",
                weight_raw=1.0,
                weight_plot=1.0
            )

    return g


def layout_bipartite_two_column(g: nx.Graph) -> Dict[str, Tuple[float, float]]:
    """
    Deterministic bipartite layout:
    - groups on left (x=0), items on right (x=1)
    - y sorted by degree so hubs sit near center
    """
    groups = [n for n, d in g.nodes(data=True) if d.get("node_type") == "group"]
    items = [n for n, d in g.nodes(data=True) if d.get("node_type") == "item"]

    groups_sorted = sorted(groups, key=lambda n: g.degree(n), reverse=True)
    items_sorted = sorted(items, key=lambda n: g.degree(n), reverse=True)

    def y_positions(nodes_sorted):
        if len(nodes_sorted) == 1:
            return {nodes_sorted[0]: 0.0}
        step = 2.0 / (len(nodes_sorted) - 1)
        return {n: 1.0 - i * step for i, n in enumerate(nodes_sorted)}

    yg = y_positions(groups_sorted)
    yi = y_positions(items_sorted)

    pos = {}
    for n in groups_sorted:
        pos[n] = (0.0, yg[n])
    for n in items_sorted:
        pos[n] = (1.0, yi[n])

    return pos


def subgraph_filter(
    g: nx.Graph,
    search: str,
    min_degree: int,
    min_weight: float,
    max_groups: int,
    largest_component_only: bool,
) -> nx.Graph:
    """Return a filtered subgraph based on UI controls."""

    # ---- Helper: choose the right edge weight key (prefer weight_plot) ----
    def _edge_w(ed: dict) -> float:
        for k in ("weight_plot", "weight_raw", "weight"):
            if k in ed and ed[k] is not None:
                try:
                    return float(ed[k])
                except Exception:
                    pass
        return 1.0

    # Edge weight filter
    edges_keep = [(u, v) for u, v, ed in g.edges(data=True) if _edge_w(ed) >= min_weight]
    sg = g.edge_subgraph(edges_keep).copy()

    # Degree filter
    nodes_keep = [n for n in sg.nodes() if sg.degree(n) >= min_degree]
    sg = sg.subgraph(nodes_keep).copy()

    # Search (keep matches + their neighbors)
    s = (search or "").strip().lower()
    if s:
        hits = [n for n, d in sg.nodes(data=True) if s in str(d.get("label", "")).lower()]
        expanded = set(hits)
        for n in hits:
            expanded.update(list(sg.neighbors(n)))
        sg = sg.subgraph(list(expanded)).copy()

    # Limit number of groups shown
    if max_groups and max_groups > 0:
        groups = [(n, sg.degree(n)) for n, d in sg.nodes(data=True) if d.get("node_type") == "group"]
        groups_sorted = sorted(groups, key=lambda x: x[1], reverse=True)
        allowed_groups = set([n for n, _ in groups_sorted[:max_groups]])

        if allowed_groups:
            keep = []
            for n, d in sg.nodes(data=True):
                if d.get("node_type") == "group":
                    if n in allowed_groups:
                        keep.append(n)
                else:
                    if any(nei in allowed_groups for nei in sg.neighbors(n)):
                        keep.append(n)
            sg = sg.subgraph(keep).copy()

    # Largest connected component only
    if largest_component_only and sg.number_of_nodes() > 0:
        comps = list(nx.connected_components(sg))
        if comps:
            biggest = max(comps, key=len)
            sg = sg.subgraph(list(biggest)).copy()

    return sg


import math
from typing import Dict, Tuple
import networkx as nx
import plotly.graph_objects as go

import math
from typing import Dict, Tuple
import networkx as nx
import plotly.graph_objects as go


def make_plotly_network(
    g: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    show_labels: bool,
    thickness_by_weight: bool = False,
    edge_width_range: Tuple[float, float] = (1.5, 6.0),
    edge_weight_range: Tuple[float, float] = (0.0, 10.0),
    highlight_nodes: dict | None = None,
) -> go.Figure:
    """
    Draw a bipartite-ish network with:
      - edges colored by term (group node)
      - optional edge thickness scaling by weight
      - optional candidate highlighting from the Insights tab
      - automatic conversion of p-values to -log10(p) when needed

    Weight handling:
      Prefer edge attr: weight_plot -> weight -> weight_raw
      If the chosen value looks like a p-value (0<val<=1), convert to -log10(val).
    """

    # ----------------------------
    # Helper: pick best available weight
    # ----------------------------
    def get_edge_weight(ed: dict) -> float:
        """
        Prefer weight_plot (already -log10(padj) ideally).
        Otherwise fall back to weight, then weight_raw.
        If it looks like a p-value (0<val<=1), convert to -log10(val).
        """
        if "weight_plot" in ed and ed["weight_plot"] is not None:
            val = ed["weight_plot"]
        elif "weight" in ed and ed["weight"] is not None:
            val = ed["weight"]
        elif "weight_raw" in ed and ed["weight_raw"] is not None:
            val = ed["weight_raw"]
        else:
            return 1.0

        try:
            val = float(val)
        except Exception:
            return 1.0

        # Auto-convert p-values to -log10(p)
        if 0.0 < val <= 1.0:
            val = max(val, 1e-300)
            return max(0.0, -math.log10(val))

        return float(val)

    # ----------------------------
    # Group edges by term (group node), keep weights
    # ----------------------------
    term_edges: dict = {}
    all_w: list[float] = []

    for u, v, ed in g.edges(data=True):
        if g.nodes[u].get("node_type") == "group":
            term = u
            gene = v
        else:
            term = v
            gene = u

        w = get_edge_weight(ed)
        term_edges.setdefault(term, []).append((term, gene, w))
        all_w.append(w)

    term_colors = term_color_map(term_edges.keys())

    # ----------------------------
    # Thickness scaling (percentile-based so small differences still show)
    # ----------------------------
    min_px, max_px = edge_width_range

    # User slider range (kept, but we also do robust scaling from the data)
    min_w_user, max_w_user = edge_weight_range
    user_has_valid_range = (max_w_user > min_w_user)

    if thickness_by_weight and all_w:
        w_arr = np.asarray(all_w, dtype=float)

        # Robust range: percentiles prevent 1 extreme edge from dominating
        lo_data, hi_data = np.percentile(w_arr, [5, 95])
        if hi_data <= lo_data:
            lo_data, hi_data = float(w_arr.min()), float(w_arr.max())

        # Optional: incorporate user range if they gave a sane one
        # (We intersect user range with data range to keep behavior intuitive.)
        if user_has_valid_range:
            lo = max(float(lo_data), float(min_w_user))
            hi = min(float(hi_data), float(max_w_user))
            if hi <= lo:
                lo, hi = float(lo_data), float(hi_data)
        else:
            lo, hi = float(lo_data), float(hi_data)

        def width_from_weight(weight: float) -> float:
            w = float(weight)

            # Clamp to robust range
            if hi > lo:
                w = max(lo, min(hi, w))
                t = (w - lo) / (hi - lo)
            else:
                t = 0.5

            # Contrast boost:
            # gamma < 1 makes small differences *more* visible (what you want for p-values)
            gamma = 0.55
            t = t ** gamma

            return float(min_px + t * (max_px - min_px))

    else:
        def width_from_weight(weight: float) -> float:
            return float(min_px)

    # ----------------------------
    # Edges
    # ----------------------------
    edge_traces = []

    for term, edges in term_edges.items():
        if thickness_by_weight:
            # One trace per edge (Plotly can't vary width inside a single trace)
            for tnode, gnode, w in edges:
                x0, y0 = pos[tnode]
                x1, y1 = pos[gnode]
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        hoverinfo="none",
                        line=dict(
                            width=width_from_weight(w),
                            color=term_colors[term],
                        ),
                        opacity=0.65,
                        showlegend=False,
                    )
                )
        else:
            # Fast mode: one trace per term, constant width
            ex, ey = [], []
            for tnode, gnode, _w in edges:
                x0, y0 = pos[tnode]
                x1, y1 = pos[gnode]
                ex += [x0, x1, None]
                ey += [y0, y1, None]

            edge_traces.append(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    hoverinfo="none",
                    line=dict(
                        width=float(min_px),
                        color=term_colors[term],
                    ),
                    opacity=0.65,
                    showlegend=False,
                )
            )

    # ----------------------------
    # Nodes
    # ----------------------------
    highlight_nodes = highlight_nodes or {}
    node_x, node_y, node_text, node_hover, node_size, node_color = [], [], [], [], [], []

    for n, d in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        label = str(d.get("label", n))
        ntype = d.get("node_type", "unknown")
        deg = g.degree(n)

        if n in highlight_nodes:
            h = highlight_nodes[n]
            rank = h.get("candidate_rank", "")
            score = h.get("followup_score", "")
            node_hover.append(
                f"⭐ Candidate #{rank}: {label}"
                f"<br>type={ntype}"
                f"<br>degree={deg}"
                f"<br>follow-up score={score}"
            )
            node_text.append(f"★ {rank}. {label}")
            node_size.append(22 + max(0, 12 - int(rank or 12)))
            node_color.append("rgba(255,193,7,0.98)")
        else:
            node_hover.append(f"{label}<br>type={ntype}<br>degree={deg}")
            node_text.append(label if show_labels else "")
            node_size.append(8 + min(24, deg * 2))

            if ntype == "group":
                node_color.append("rgba(0,119,182,0.95)")
            else:
                node_color.append("rgba(0,180,216,0.95)")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(
                width=[3 if n in highlight_nodes else 1 for n in g.nodes()],
                color=["rgba(120,53,15,0.95)" if n in highlight_nodes else "rgba(0,0,0,0.25)" for n in g.nodes()],
            ),
        ),
        name="nodes",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan",
        hovermode="closest",
    )

    return fig


def graph_stats(g: nx.Graph) -> Dict[str, object]:
    if g.number_of_nodes() == 0:
        return dict(
            nodes=0, edges=0, items=0, groups=0, components=0, largest_component=0,
            top_groups=[], top_items=[],
            weight_key=None,
            weight_summary=None,
            top_weighted_edges=[]
        )

    items = [n for n, d in g.nodes(data=True) if d.get("node_type") == "item"]
    groups = [n for n, d in g.nodes(data=True) if d.get("node_type") == "group"]

    comps = list(nx.connected_components(g))
    largest = max((len(c) for c in comps), default=0)

    top_groups = sorted(
        [(g.nodes[n].get("label", n), g.degree(n)) for n in groups],
        key=lambda x: x[1], reverse=True
    )[:20]

    top_items = sorted(
        [(g.nodes[n].get("label", n), g.degree(n)) for n in items],
        key=lambda x: x[1], reverse=True
    )[:20]

    # ----------------------------
    # Edge weight stats (if present)
    # Prefer weight_plot > weight > weight_raw
    # ----------------------------
    def pick_weight(ed: dict):
        for k in ("weight_plot", "weight", "weight_raw"):
            if k in ed and ed[k] is not None:
                try:
                    return k, float(ed[k])
                except Exception:
                    pass
        return None, None

    weights = []
    chosen_key = None

    for u, v, ed in g.edges(data=True):
        k, w = pick_weight(ed)
        if w is None:
            continue
        if chosen_key is None:
            chosen_key = k
        # If some edges have multiple keys, keep using the preferred order:
        # Only accept values that match the chosen key.
        if k == chosen_key:
            weights.append(w)

    weight_summary = None
    top_weighted_edges = []

    if weights:
        arr = np.array(weights, dtype=float)

        weight_summary = dict(
            key=chosen_key,
            min=float(np.min(arr)),
            p25=float(np.percentile(arr, 25)),
            median=float(np.median(arr)),
            mean=float(np.mean(arr)),
            p75=float(np.percentile(arr, 75)),
            max=float(np.max(arr)),
        )

        # Top weighted edges (show endpoints + labels + weight)
        # Note: only include edges that actually have the chosen_key
        edges_with_w = []
        for u, v, ed in g.edges(data=True):
            if chosen_key in ed and ed[chosen_key] is not None:
                try:
                    w = float(ed[chosen_key])
                except Exception:
                    continue
                u_lab = g.nodes[u].get("label", u)
                v_lab = g.nodes[v].get("label", v)
                edges_with_w.append((u_lab, v_lab, w))

        edges_with_w.sort(key=lambda x: x[2], reverse=True)
        top_weighted_edges = edges_with_w[:20]

    return dict(
        nodes=g.number_of_nodes(),
        edges=g.number_of_edges(),
        items=len(items),
        groups=len(groups),
        components=len(comps),
        largest_component=largest,
        top_groups=top_groups,
        top_items=top_items,
        weight_key=chosen_key,
        weight_summary=weight_summary,
        top_weighted_edges=top_weighted_edges,
    )


# ----------------------------
# Large graph warning helpers
# ----------------------------
LARGE_GRAPH_NODE_WARNING = 1500
LARGE_GRAPH_EDGE_WARNING = 5000
VERY_LARGE_RENDER_NODE_LIMIT = 1200
VERY_LARGE_RENDER_EDGE_LIMIT = 4000


def graph_size_warning_component(node_count: int, edge_count: int, context: str = "network"):
    """Return a small UI warning for large graphs, or an empty string for normal sizes."""
    if node_count < LARGE_GRAPH_NODE_WARNING and edge_count < LARGE_GRAPH_EDGE_WARNING:
        return ""

    return html.Div(
        style={
            "border": "1px solid #fed7aa",
            "borderLeft": "5px solid #f97316",
            "borderRadius": "10px",
            "padding": "10px",
            "background": "#fff7ed",
            "color": "#7c2d12",
            "fontSize": "0.86rem",
            "lineHeight": "1.35",
            "marginTop": "10px",
        },
        children=[
            html.Div("Large graph warning", style={"fontWeight": "800", "marginBottom": "4px"}),
            html.Div(
                f"The current {context} has {node_count:,} nodes and {edge_count:,} edges. "
                "Rendering may be slow."
            ),
            html.Div(
                "Try increasing minimum node degree, increasing minimum edge weight, reducing maximum groups, "
                "using search to focus on a pathway family, or switching away from force-directed layout.",
                style={"marginTop": "4px"},
            ),
        ],
    )


def large_graph_placeholder_figure(node_count: int, edge_count: int) -> go.Figure:
    """Avoid trying to render extremely large filtered graphs in the browser."""
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text=(
                    "Large filtered graph skipped for browser performance.<br>"
                    f"Filtered graph: {node_count:,} nodes, {edge_count:,} edges.<br>"
                    "Use min degree, min edge weight, max groups, or search filters to reduce the graph."
                ),
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                font=dict(size=16),
                align="center",
            )
        ],
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig



def rebuild_graph_from_store(graph_data: dict) -> nx.Graph:
    """Rebuild a NetworkX graph from the serialized Dash store."""
    g = nx.Graph()

    if not graph_data:
        return g

    for nd in graph_data.get("nodes", []):
        node_id = nd.get("id")
        if not node_id:
            continue
        attrs = nd.copy()
        attrs.pop("id", None)
        g.add_node(node_id, **attrs)

    for ed in graph_data.get("edges", []):
        u, v = ed.get("source"), ed.get("target")
        if not u or not v:
            continue
        attrs = ed.copy()
        attrs.pop("source", None)
        attrs.pop("target", None)
        g.add_edge(u, v, **attrs)

    return g


def _normalized_edge_weight(ed: dict) -> float:
    """
    Return a positive edge weight suitable for random-walk analysis.

    Prefer weight_plot because this app already stores adjusted p-values as
    -log10(adjusted p-value). Fall back to weight, then weight_raw.
    If a fallback looks like a p-value, convert it to -log10(p).
    """
    val = None
    for key in ("weight_plot", "weight", "weight_raw"):
        if key in ed and ed[key] is not None:
            val = ed[key]
            break

    if val is None:
        return 1.0

    try:
        val = float(val)
    except Exception:
        return 1.0

    if 0.0 < val <= 1.0:
        val = max(val, 1e-300)
        val = -math.log10(val)

    # PageRank needs non-negative weights. A tiny floor avoids all-zero issues.
    return max(float(val), 1e-12)


def prepare_markov_graph(g: nx.Graph, ranking_mode: str = "balanced") -> nx.Graph:
    """
    Copy the current graph and add a 'markov_weight' edge attribute.

    This does not alter the visualization graph. It only prepares a separate
    weighted graph for signal diffusion / random-walk analysis.

    Ranking modes:
      - evidence: edge weights are driven by -log10(adjusted p-value)
      - connectivity: edge weights are treated equally so topology dominates
      - balanced: dampens extreme p-value effects while preserving evidence
    """
    mode = (ranking_mode or "balanced").lower().strip()
    mg = g.copy()

    for u, v, ed in mg.edges(data=True):
        evidence_w = _normalized_edge_weight(ed)

        if mode == "evidence":
            markov_w = evidence_w
        elif mode == "connectivity":
            markov_w = 1.0
        else:
            # Balanced mode keeps statistical evidence but compresses extreme
            # single-edge p-value effects so connected pathway structure matters.
            markov_w = math.sqrt(evidence_w)

        ed["markov_weight"] = max(float(markov_w), 1e-12)

    return mg


def _degree_adjusted_score(g: nx.Graph, node_id: str, pagerank_score: float, ranking_mode: str) -> float:
    """
    Convert the raw random-walk score into a user-facing priority score.

    Raw PageRank can still reward a strong isolated edge. For enrichment
    interpretation, especially pathway prioritization, degree should matter:
    pathways supported by multiple genes/items should not be buried by a
    one-edge term with an extremely strong p-value.
    """
    mode = (ranking_mode or "balanced").lower().strip()
    degree = max(int(g.degree(node_id)), 1)

    if mode == "evidence":
        # Pure evidence mode stays closest to classic weighted PageRank.
        degree_factor = 1.0
    elif mode == "connectivity":
        # Connectivity mode strongly rewards shared-network structure.
        degree_factor = 1.0 + math.log1p(degree)
    else:
        # Balanced mode gives moderate support to connected pathways while
        # avoiding the "everything is just degree" problem.
        degree_factor = math.sqrt(1.0 + math.log1p(degree))

    return float(pagerank_score) * float(degree_factor)


def run_signal_diffusion(
    g: nx.Graph,
    seed_node: str | None = None,
    alpha: float = 0.85,
    ranking_mode: str = "balanced",
) -> dict:
    """
    Run weighted PageRank / personalized PageRank over the enrichment network.

    Unseeded mode asks: which nodes are globally influential in this network?
    Seeded mode asks: starting from one selected gene/pathway, which nodes are
    most reachable through weighted network diffusion?

    ranking_mode controls how strongly evidence vs connectivity affects the walk.
    """
    if g.number_of_nodes() == 0:
        return {}

    mg = prepare_markov_graph(g, ranking_mode=ranking_mode)
    personalization = None

    if seed_node and seed_node in mg.nodes:
        personalization = {n: 0.0 for n in mg.nodes}
        personalization[seed_node] = 1.0

    try:
        scores = nx.pagerank(
            mg,
            alpha=float(alpha),
            weight="markov_weight",
            personalization=personalization,
            max_iter=500,
            tol=1e-10,
        )
    except nx.PowerIterationFailedConvergence:
        scores = nx.pagerank(
            mg,
            alpha=float(alpha),
            weight="markov_weight",
            personalization=personalization,
            max_iter=2000,
            tol=1e-8,
        )

    return scores


def diffusion_rows(
    g: nx.Graph,
    scores: dict,
    top_n: int = 50,
    ranking_mode: str = "balanced",
) -> list[dict]:
    """Convert diffusion scores into table-friendly rows."""
    rows = []
    for node_id, score in scores.items():
        node_attrs = g.nodes[node_id]
        priority_score = _degree_adjusted_score(g, node_id, float(score), ranking_mode)
        rows.append(
            {
                "label": node_attrs.get("label", node_id),
                "node_type": node_attrs.get("node_type", "unknown"),
                "degree": int(g.degree(node_id)),
                "priority_score": round(float(priority_score), 8),
                "raw_diffusion_score": round(float(score), 8),
                "diffusion_score": round(float(priority_score), 8),
                "ranking_mode": ranking_mode,
                "node_id": node_id,
            }
        )

    rows = sorted(rows, key=lambda r: r["priority_score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return rows[: int(top_n)]


def split_diffusion_rows(rows: list[dict], top_n: int = 25) -> tuple[list[dict], list[dict]]:
    """Split diffusion results into pathway/group and gene/item result tables."""
    group_rows = [r for r in rows if r.get("node_type") == "group"][: int(top_n)]
    item_rows = [r for r in rows if r.get("node_type") == "item"][: int(top_n)]
    return group_rows, item_rows


def _minmax01(value: float, values: list[float]) -> float:
    """Scale a value to 0..1 using the current result set."""
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return 0.0
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return 1.0
    return float((float(value) - lo) / (hi - lo))


def _incident_evidence_summary(g: nx.Graph, node_id: str) -> dict:
    """Summarize direct statistical evidence around a node from incident edges."""
    weights = []
    raw_pvals = []

    for _u, _v, ed in g.edges(node_id, data=True):
        weights.append(_normalized_edge_weight(ed))
        if "weight_raw" in ed and ed["weight_raw"] is not None:
            try:
                raw = float(ed["weight_raw"])
                if raw > 0:
                    raw_pvals.append(raw)
            except Exception:
                pass

    if not weights:
        return {
            "max_evidence": 0.0,
            "mean_evidence": 0.0,
            "best_adj_p": None,
        }

    return {
        "max_evidence": float(max(weights)),
        "mean_evidence": float(np.mean(weights)),
        "best_adj_p": float(min(raw_pvals)) if raw_pvals else None,
    }


def candidate_rows(
    g: nx.Graph,
    diffusion_result_rows: list[dict],
    top_n: int = 30,
    node_type: str = "group",
) -> list[dict]:
    """
    Build a follow-up candidate table that combines statistical evidence,
    network diffusion, and support breadth.

    This is meant to answer: what should a scientist inspect first?
    It is not a new p-value. It is a pragmatic prioritization score.
    """
    rows = [r for r in diffusion_result_rows if r.get("node_type") == node_type]
    if not rows:
        return []

    enriched = []
    for r in rows:
        node_id = r.get("node_id")
        if node_id not in g.nodes:
            continue
        ev = _incident_evidence_summary(g, node_id)
        enriched.append({**r, **ev})

    if not enriched:
        return []

    priority_values = [float(r.get("priority_score", 0.0)) for r in enriched]
    evidence_values = [float(r.get("max_evidence", 0.0)) for r in enriched]
    degree_values = [float(r.get("degree", 0.0)) for r in enriched]

    out = []
    for r in enriched:
        diffusion_component = _minmax01(float(r.get("priority_score", 0.0)), priority_values)
        evidence_component = _minmax01(float(r.get("max_evidence", 0.0)), evidence_values)
        support_component = _minmax01(float(r.get("degree", 0.0)), degree_values)

        # Default product-facing score: diffusion is most important, direct evidence
        # keeps it grounded in enrichment strength, and support breadth prevents one-edge
        # terms from dominating when broader biology is present.
        followup_score = (
            0.50 * diffusion_component
            + 0.30 * evidence_component
            + 0.20 * support_component
        )

        best_adj_p = r.get("best_adj_p")
        out.append({
            "label": r.get("label"),
            "node_type": r.get("node_type"),
            "followup_score": round(float(followup_score), 8),
            "priority_score": r.get("priority_score"),
            "raw_diffusion_score": r.get("raw_diffusion_score"),
            "max_evidence": round(float(r.get("max_evidence", 0.0)), 6),
            "mean_evidence": round(float(r.get("mean_evidence", 0.0)), 6),
            "best_adj_p": None if best_adj_p is None else float(best_adj_p),
            "degree": r.get("degree"),
            "ranking_mode": r.get("ranking_mode"),
            "node_id": r.get("node_id"),
        })

    out = sorted(out, key=lambda r: r["followup_score"], reverse=True)[: int(top_n)]
    for i, row in enumerate(out, start=1):
        row["candidate_rank"] = i
    return out



# ----------------------------
# Pathway projection helpers
# ----------------------------

def build_pathway_projection_graph(g: nx.Graph, method: str = "jaccard") -> nx.Graph:
    """Build a pathway-only graph: pathways connect when they share genes/items."""
    method = method or "jaccard"
    pg = nx.Graph()
    groups = [n for n, d in g.nodes(data=True) if d.get("node_type") == "group"]
    items = [n for n, d in g.nodes(data=True) if d.get("node_type") == "item"]

    for gr in groups:
        attrs = g.nodes[gr].copy()
        pg.add_node(
            gr,
            label=attrs.get("label", gr),
            node_type="projected_group",
            source_degree=int(g.degree(gr)),
        )

    group_items = {
        gr: set(nei for nei in g.neighbors(gr) if g.nodes[nei].get("node_type") == "item")
        for gr in groups
    }

    item_groups = {}
    for item in items:
        connected_groups = [nei for nei in g.neighbors(item) if g.nodes[nei].get("node_type") == "group"]
        if len(connected_groups) >= 2:
            item_groups[item] = connected_groups

    pair_payload = {}
    for item, connected_groups in item_groups.items():
        for a, b in combinations(sorted(connected_groups), 2):
            key = (a, b)
            if key not in pair_payload:
                pair_payload[key] = {"shared_items": [], "weighted_support": 0.0}
            wa = _normalized_edge_weight(g.edges[a, item]) if g.has_edge(a, item) else 1.0
            wb = _normalized_edge_weight(g.edges[b, item]) if g.has_edge(b, item) else 1.0
            pair_payload[key]["shared_items"].append(g.nodes[item].get("label", item))
            pair_payload[key]["weighted_support"] += (wa + wb) / 2.0

    for (a, b), payload in pair_payload.items():
        shared_count = len(payload["shared_items"])
        union_count = len(group_items.get(a, set()) | group_items.get(b, set()))
        jaccard = (shared_count / union_count) if union_count else 0.0
        weighted_shared = float(payload["weighted_support"])

        if method == "shared_count":
            edge_weight = float(shared_count)
        elif method == "weighted_shared":
            edge_weight = weighted_shared
        else:
            edge_weight = float(jaccard)

        pg.add_edge(
            a,
            b,
            weight=float(edge_weight),
            shared_count=int(shared_count),
            jaccard=float(jaccard),
            weighted_shared=float(weighted_shared),
            shared_items="; ".join(map(str, payload["shared_items"][:50])),
        )

    isolated = [n for n in pg.nodes if pg.degree(n) == 0]
    pg.remove_nodes_from(isolated)
    return pg


def projection_graph_to_store(pg: nx.Graph) -> dict:
    nodes = [{"id": n, **d} for n, d in pg.nodes(data=True)]
    edges = [{"source": u, "target": v, **ed} for u, v, ed in pg.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


def rebuild_projection_from_store(graph_data: dict) -> nx.Graph:
    pg = nx.Graph()
    if not graph_data:
        return pg
    for nd in graph_data.get("nodes", []):
        node_id = nd.get("id")
        if not node_id:
            continue
        attrs = nd.copy()
        attrs.pop("id", None)
        pg.add_node(node_id, **attrs)
    for ed in graph_data.get("edges", []):
        u, v = ed.get("source"), ed.get("target")
        if not u or not v:
            continue
        attrs = ed.copy()
        attrs.pop("source", None)
        attrs.pop("target", None)
        pg.add_edge(u, v, **attrs)
    return pg


def make_projection_figure(pg: nx.Graph, show_labels: bool = True) -> go.Figure:
    if pg.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="Build a projection network after building the main graph.", showarrow=False)],
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    pos = nx.spring_layout(pg, seed=11, k=1 / math.sqrt(max(1, pg.number_of_nodes())))
    weights = [float(ed.get("weight", 1.0)) for _, _, ed in pg.edges(data=True)]

    if weights:
        w_arr = np.asarray(weights, dtype=float)
        lo, hi = float(np.percentile(w_arr, 5)), float(np.percentile(w_arr, 95))
        if hi <= lo:
            lo, hi = float(w_arr.min()), float(w_arr.max())
    else:
        lo, hi = 0.0, 1.0

    def width_from_weight(w: float) -> float:
        if hi > lo:
            t = max(0.0, min(1.0, (float(w) - lo) / (hi - lo)))
        else:
            t = 0.5
        return 1.0 + t * 5.0

    edge_traces = []
    for u, v, ed in pg.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        hover = (
            f"{pg.nodes[u].get('label', u)} ↔ {pg.nodes[v].get('label', v)}"
            f"<br>weight={float(ed.get('weight', 0.0)):.4g}"
            f"<br>shared items={ed.get('shared_count', 0)}"
            f"<br>jaccard={float(ed.get('jaccard', 0.0)):.4g}"
        )
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                hovertext=hover,
                hoverinfo="text",
                line=dict(width=width_from_weight(ed.get("weight", 1.0)), color="rgba(100,116,139,0.55)"),
                showlegend=False,
            )
        )

    pr = nx.pagerank(pg, weight="weight") if pg.number_of_edges() else {
        n: 1 / max(1, pg.number_of_nodes()) for n in pg.nodes
    }
    max_pr = max(pr.values()) if pr else 1.0

    node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
    for n, d in pg.nodes(data=True):
        x, y = pos[n]
        label = str(d.get("label", n))
        deg = pg.degree(n)
        node_x.append(x)
        node_y.append(y)
        node_text.append(label if show_labels else "")
        node_hover.append(
            f"{label}<br>projection degree={deg}"
            f"<br>source membership degree={d.get('source_degree', '—')}"
            f"<br>PageRank={pr.get(n, 0.0):.6f}"
        )
        node_size.append(12 + 26 * (pr.get(n, 0.0) / max_pr if max_pr else 0.0))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color="rgba(0,119,182,0.92)",
            line=dict(width=1.5, color="rgba(0,0,0,0.35)"),
        ),
        name="pathways",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan",
        hovermode="closest",
    )
    return fig


def projection_stats_rows(pg: nx.Graph, top_n: int = 30) -> list[dict]:
    if pg.number_of_nodes() == 0:
        return []
    pr = nx.pagerank(pg, weight="weight") if pg.number_of_edges() else {n: 1 / pg.number_of_nodes() for n in pg.nodes}
    rows = []
    for n, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            "rank": len(rows) + 1,
            "pathway": pg.nodes[n].get("label", n),
            "projection_score": round(float(score), 8),
            "projection_degree": int(pg.degree(n)),
            "source_degree": int(pg.nodes[n].get("source_degree", 0)),
            "node_id": n,
        })
    return rows[:int(top_n)]


def prepare_projection_markov_graph(pg: nx.Graph, ranking_mode: str = "balanced") -> nx.Graph:
    mg = pg.copy()
    mode = (ranking_mode or "balanced").lower().strip()
    for _u, _v, ed in mg.edges(data=True):
        try:
            overlap_w = float(ed.get("weight", 1.0))
        except Exception:
            overlap_w = 1.0
        overlap_w = max(overlap_w, 1e-12)
        if mode == "connectivity":
            markov_w = 1.0
        elif mode == "evidence":
            markov_w = overlap_w
        else:
            markov_w = math.sqrt(overlap_w)
        ed["markov_weight"] = max(float(markov_w), 1e-12)
    return mg


def run_projection_diffusion(pg: nx.Graph, seed_node: str | None = None, alpha: float = 0.85, ranking_mode: str = "balanced") -> dict:
    if pg.number_of_nodes() == 0:
        return {}
    mg = prepare_projection_markov_graph(pg, ranking_mode=ranking_mode)
    personalization = None
    if seed_node and seed_node in mg.nodes:
        personalization = {n: 0.0 for n in mg.nodes}
        personalization[seed_node] = 1.0
    try:
        return nx.pagerank(mg, alpha=float(alpha), weight="markov_weight", personalization=personalization, max_iter=500, tol=1e-10)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(mg, alpha=float(alpha), weight="markov_weight", personalization=personalization, max_iter=2000, tol=1e-8)


def _projection_degree_adjusted_score(pg: nx.Graph, node_id: str, pagerank_score: float, ranking_mode: str) -> float:
    mode = (ranking_mode or "balanced").lower().strip()
    degree = max(int(pg.degree(node_id)), 1)
    if mode == "evidence":
        degree_factor = 1.0
    elif mode == "connectivity":
        degree_factor = 1.0 + math.log1p(degree)
    else:
        degree_factor = math.sqrt(1.0 + math.log1p(degree))
    return float(pagerank_score) * float(degree_factor)


def projection_diffusion_rows(pg: nx.Graph, scores: dict, top_n: int = 50, ranking_mode: str = "balanced") -> list[dict]:
    rows = []
    for node_id, score in scores.items():
        if node_id not in pg.nodes:
            continue
        attrs = pg.nodes[node_id]
        priority_score = _projection_degree_adjusted_score(pg, node_id, float(score), ranking_mode)
        rows.append({
            "label": attrs.get("label", node_id),
            "node_type": "projected_group",
            "degree": int(pg.degree(node_id)),
            "source_degree": int(attrs.get("source_degree", 0)),
            "priority_score": round(float(priority_score), 8),
            "raw_diffusion_score": round(float(score), 8),
            "diffusion_score": round(float(priority_score), 8),
            "ranking_mode": ranking_mode,
            "node_id": node_id,
        })
    rows = sorted(rows, key=lambda r: r["priority_score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows[: int(top_n)]


def _projection_overlap_summary(pg: nx.Graph, node_id: str) -> dict:
    weights, shared_counts = [], []
    for _u, _v, ed in pg.edges(node_id, data=True):
        try:
            weights.append(float(ed.get("weight", 0.0)))
        except Exception:
            pass
        try:
            shared_counts.append(float(ed.get("shared_count", 0.0)))
        except Exception:
            pass
    if not weights:
        return {"mean_overlap_weight": 0.0, "max_overlap_weight": 0.0, "mean_shared_count": 0.0, "max_shared_count": 0.0}
    return {
        "mean_overlap_weight": float(np.mean(weights)),
        "max_overlap_weight": float(max(weights)),
        "mean_shared_count": float(np.mean(shared_counts)) if shared_counts else 0.0,
        "max_shared_count": float(max(shared_counts)) if shared_counts else 0.0,
    }


def projection_candidate_rows(pg: nx.Graph, diffusion_result_rows: list[dict], top_n: int = 30) -> list[dict]:
    if not diffusion_result_rows:
        return []
    enriched = []
    for r in diffusion_result_rows:
        node_id = r.get("node_id")
        if node_id not in pg.nodes:
            continue
        enriched.append({**r, **_projection_overlap_summary(pg, node_id)})
    if not enriched:
        return []
    priority_values = [float(r.get("priority_score", 0.0)) for r in enriched]
    degree_values = [float(r.get("degree", 0.0)) for r in enriched]
    overlap_values = [float(r.get("mean_overlap_weight", 0.0)) for r in enriched]
    source_degree_values = [float(r.get("source_degree", 0.0)) for r in enriched]
    out = []
    for r in enriched:
        followup_score = (0.50 * _minmax01(float(r.get("priority_score", 0.0)), priority_values)
                          + 0.25 * _minmax01(float(r.get("degree", 0.0)), degree_values)
                          + 0.15 * _minmax01(float(r.get("mean_overlap_weight", 0.0)), overlap_values)
                          + 0.10 * _minmax01(float(r.get("source_degree", 0.0)), source_degree_values))
        out.append({
            "label": r.get("label"),
            "node_type": r.get("node_type"),
            "followup_score": round(float(followup_score), 8),
            "priority_score": r.get("priority_score"),
            "raw_diffusion_score": r.get("raw_diffusion_score"),
            "degree": r.get("degree"),
            "source_degree": r.get("source_degree"),
            "mean_overlap_weight": round(float(r.get("mean_overlap_weight", 0.0)), 6),
            "max_overlap_weight": round(float(r.get("max_overlap_weight", 0.0)), 6),
            "mean_shared_count": round(float(r.get("mean_shared_count", 0.0)), 6),
            "max_shared_count": round(float(r.get("max_shared_count", 0.0)), 6),
            "ranking_mode": r.get("ranking_mode"),
            "node_id": r.get("node_id"),
        })
    out = sorted(out, key=lambda r: r["followup_score"], reverse=True)[: int(top_n)]
    for i, row in enumerate(out, start=1):
        row["candidate_rank"] = i
    return out


# ----------------------------
# Consensus candidate helpers
# ----------------------------

def consensus_candidate_rows(bipartite_store: dict | None, projection_store: dict | None, top_n: int = 30) -> list[dict]:
    """Combine bipartite and projection candidate scores into one final pathway list."""
    if not bipartite_store or not projection_store:
        return []

    bip_rows = bipartite_store.get("candidate_rows") or []
    proj_rows = projection_store.get("candidate_rows") or []

    bip_by_key = {}
    for r in bip_rows:
        key = r.get("node_id") or r.get("label")
        if key:
            bip_by_key[key] = r

    proj_by_key = {}
    for r in proj_rows:
        key = r.get("node_id") or r.get("label")
        if key:
            proj_by_key[key] = r

    keys = sorted(set(bip_by_key) | set(proj_by_key))
    if not keys:
        return []

    bip_scores = [float(r.get("followup_score", 0.0)) for r in bip_by_key.values()]
    proj_scores = [float(r.get("followup_score", 0.0)) for r in proj_by_key.values()]

    out = []
    for key in keys:
        b = bip_by_key.get(key, {})
        pr = proj_by_key.get(key, {})
        label = b.get("label") or pr.get("label") or str(key)

        b_score_raw = float(b.get("followup_score", 0.0)) if b else 0.0
        p_score_raw = float(pr.get("followup_score", 0.0)) if pr else 0.0

        b_score_norm = _minmax01(b_score_raw, bip_scores) if bip_scores else 0.0
        p_score_norm = _minmax01(p_score_raw, proj_scores) if proj_scores else 0.0
        consensus_score = 0.50 * b_score_norm + 0.50 * p_score_norm

        out.append({
            "label": label,
            "consensus_score": round(float(consensus_score), 8),
            "bipartite_followup_score": round(float(b_score_raw), 8),
            "projection_followup_score": round(float(p_score_raw), 8),
            "bipartite_candidate_rank": b.get("candidate_rank"),
            "projection_candidate_rank": pr.get("candidate_rank"),
            "bipartite_priority_score": b.get("priority_score"),
            "projection_priority_score": pr.get("priority_score"),
            "best_adj_p": b.get("best_adj_p"),
            "max_evidence": b.get("max_evidence"),
            "bipartite_degree": b.get("degree"),
            "projection_degree": pr.get("degree"),
            "source_degree": pr.get("source_degree"),
            "mean_overlap_weight": pr.get("mean_overlap_weight"),
            "max_shared_count": pr.get("max_shared_count"),
            "node_id": key,
        })

    out = sorted(out, key=lambda r: r["consensus_score"], reverse=True)[: int(top_n)]
    for i, row in enumerate(out, start=1):
        row["consensus_rank"] = i
    return out

# ----------------------------
# Dash App
# ----------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # for gunicorn


# ==========================
# MAIN PANEL (tabs)
# ==========================
main_panel = html.Div(
    style={
        "flex": "1",
        "background": "white",
        "borderRadius": "14px",
        "padding": "14px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.12)",
        "height": "calc(100vh - 32px)",
        "overflow": "auto",
    },
    children=[
        dcc.Tabs(
            id="tabs",
            value="plot",
            colors={
                "border": "#CBD5E1",
                "primary": "#1E3A8A",     # active tab underline
                "background": "#EEF2FF",  # inactive tabs
            },
            children=[
                # --------------------------
                # Plot tab
                # --------------------------
                dcc.Tab(
                    label="Plot",
                    value="plot",
                    children=[
                        html.Div(
                            id="plot-hint",
                            style={"marginTop": "10px", "color": "#6b7280", "fontSize": "0.95rem"},
                            children="Upload a raw CSV and click “Preprocess / Build graph” to begin.",
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginTop": "10px", "flexWrap": "wrap"},
                            children=[
                                html.Button(
                                    "Download main graph as SVG",
                                    id="btn-dl-main-svg",
                                    n_clicks=0,
                                    style={
                                        "padding": "9px 12px",
                                        "borderRadius": "10px",
                                        "border": "1px solid #003566",
                                        "background": "white",
                                        "color": "#003566",
                                        "fontWeight": "800",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Download main graph as PDF",
                                    id="btn-dl-main-pdf",
                                    n_clicks=0,
                                    style={
                                        "padding": "9px 12px",
                                        "borderRadius": "10px",
                                        "border": "1px solid #003566",
                                        "background": "white",
                                        "color": "#003566",
                                        "fontWeight": "800",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(
                                    "PNG is still available from the Plotly camera icon.",
                                    style={"alignSelf": "center", "color": "#6b7280", "fontSize": "0.9rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="network-graph", style={"height": "80vh", "marginTop": "10px"}),
                    ],
                ),

                # --------------------------
                # Stats tab
                # --------------------------
                dcc.Tab(
                    label="Stats",
                    value="stats",
                    children=[
                        html.Div(id="stats-cards", style={"marginTop": "12px"}),

                        html.H4("Top groups (by degree)", style={"marginTop": "14px", "color": "#003566"}),
                        dash_table.DataTable(
                            id="top-groups-table",
                            columns=[{"name": "Group", "id": "group"}, {"name": "Degree", "id": "degree"}],
                            data=[],
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px"},
                        ),

                        html.H4("Top items (by degree)", style={"marginTop": "14px", "color": "#003566"}),
                        dash_table.DataTable(
                            id="top-items-table",
                            columns=[{"name": "Item", "id": "item"}, {"name": "Degree", "id": "degree"}],
                            data=[],
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px"},
                        ),
                    ],
                ),

                # --------------------------
                # Insights tab
                # --------------------------
                dcc.Tab(
                    label="Bipartite Diffusion",
                    value="insights",
                    children=[
                        html.Div(
                            style={"maxWidth": "1100px", "padding": "10px 6px"},
                            children=[
                                html.H3("Signal Diffusion on Main Gene ↔ Pathway Graph", style={"color": "#003566"}),
                                html.P(
                                    "This feature runs weighted random-walk ranking over the main bipartite enrichment network. "
                                    "Use global mode to find broadly influential genes/pathways, or choose a seed node "
                                    "to find the terms and genes most reachable from that starting point.",
                                    style={"color": "#111827"},
                                ),

                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                                        "gap": "10px",
                                        "marginTop": "12px",
                                    },
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Label("Optional seed node"),
                                                dcc.Dropdown(
                                                    id="markov-seed-node",
                                                    options=[],
                                                    placeholder="Global ranking — no seed selected",
                                                    clearable=True,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Walk retention / alpha"),
                                                dcc.Slider(
                                                    id="markov-alpha",
                                                    min=0.50,
                                                    max=0.95,
                                                    step=0.05,
                                                    value=0.85,
                                                    marks={0.50: "0.50", 0.85: "0.85", 0.95: "0.95"},
                                                    tooltip={"placement": "bottom"},
                                                    updatemode="mouseup",
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Ranking mode"),
                                                dcc.Dropdown(
                                                    id="markov-ranking-mode",
                                                    options=[
                                                        {"label": "Balanced — evidence + connectivity", "value": "balanced"},
                                                        {"label": "Evidence-weighted — strong adjusted p-values", "value": "evidence"},
                                                        {"label": "Connectivity-weighted — network hubs/shared biology", "value": "connectivity"},
                                                    ],
                                                    value="balanced",
                                                    clearable=False,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Rows per table"),
                                                dcc.Slider(
                                                    id="markov-top-n",
                                                    min=10,
                                                    max=100,
                                                    step=10,
                                                    value=30,
                                                    marks={10: "10", 50: "50", 100: "100"},
                                                    tooltip={"placement": "bottom"},
                                                    updatemode="mouseup",
                                                ),
                                            ]
                                        ),
                                    ],
                                ),

                                html.Button(
                                    "Run Signal Diffusion Analysis",
                                    id="btn-run-markov",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "14px",
                                        "padding": "10px 14px",
                                        "borderRadius": "10px",
                                        "border": "none",
                                        "background": "#003566",
                                        "color": "white",
                                        "fontWeight": "800",
                                        "cursor": "pointer",
                                    },
                                ),

                                html.Div(id="markov-status", style={"marginTop": "10px", "color": "#4b5563"}),
                                html.Div(id="markov-summary-cards", style={"marginTop": "12px"}),

                                dcc.Tabs(
                                    id="diffusion-subtabs",
                                    value="candidate-tab",
                                    colors={
                                        "border": "#CBD5E1",
                                        "primary": "#0077b6",
                                        "background": "#F8FAFC",
                                    },
                                    style={"marginTop": "16px"},
                                    children=[
                                        dcc.Tab(
                                            label="Top Candidates",
                                            value="candidate-tab",
                                            children=[
                                                html.Div(
                                                    style={"padding": "12px 4px"},
                                                    children=[
                                                        html.H4("Top follow-up candidates — pathways / terms", style={"marginTop": "4px", "color": "#003566"}),
                                                        html.P(
                                                            "This table combines diffusion priority, direct enrichment evidence, and support breadth. "
                                                            "Use it as the most practical first-pass list for biological follow-up.",
                                                            style={"color": "#4b5563", "fontSize": "0.92rem"},
                                                        ),
                                                        html.Button(
                                                            "Download top_candidates.csv",
                                                            id="btn-dl-candidates",
                                                            n_clicks=0,
                                                            style={
                                                                "marginBottom": "10px",
                                                                "padding": "9px 12px",
                                                                "borderRadius": "10px",
                                                                "border": "1px solid #003566",
                                                                "background": "white",
                                                                "color": "#003566",
                                                                "fontWeight": "800",
                                                                "cursor": "pointer",
                                                            },
                                                        ),
                                                        dash_table.DataTable(
                                                            id="markov-candidate-pathways-table",
                                                            columns=[
                                                                {"name": "Candidate rank", "id": "candidate_rank"},
                                                                {"name": "Pathway / term", "id": "label"},
                                                                {"name": "Follow-up score", "id": "followup_score"},
                                                                {"name": "Diffusion priority", "id": "priority_score"},
                                                                {"name": "Max evidence (-log10 adj p)", "id": "max_evidence"},
                                                                {"name": "Best adj p", "id": "best_adj_p"},
                                                                {"name": "Degree", "id": "degree"},
                                                            ],
                                                            data=[],
                                                            page_size=15,
                                                            sort_action="native",
                                                            style_table={"overflowX": "auto"},
                                                            style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"},
                                                        ),
                                                    ],
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Diffusion Rankings",
                                            value="rankings-tab",
                                            children=[
                                                html.Div(
                                                    style={"padding": "12px 4px"},
                                                    children=[
                                                        html.Button(
                                                            "Download diffusion_results.csv",
                                                            id="btn-dl-markov",
                                                            n_clicks=0,
                                                            style={
                                                                "marginBottom": "10px",
                                                                "padding": "9px 12px",
                                                                "borderRadius": "10px",
                                                                "border": "1px solid #003566",
                                                                "background": "white",
                                                                "color": "#003566",
                                                                "fontWeight": "800",
                                                                "cursor": "pointer",
                                                            },
                                                        ),
                                                        html.H4("Top pathways / terms by diffusion score", style={"marginTop": "4px", "color": "#003566"}),
                                                        dash_table.DataTable(
                                                            id="markov-top-groups-table",
                                                            columns=[
                                                                {"name": "Rank", "id": "rank"},
                                                                {"name": "Pathway / term", "id": "label"},
                                                                {"name": "Priority score", "id": "priority_score"},
                                                                {"name": "Raw diffusion", "id": "raw_diffusion_score"},
                                                                {"name": "Degree", "id": "degree"},
                                                            ],
                                                            data=[],
                                                            page_size=15,
                                                            sort_action="native",
                                                            style_table={"overflowX": "auto"},
                                                            style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"},
                                                        ),

                                                        html.H4("Top genes / items by diffusion score", style={"marginTop": "16px", "color": "#003566"}),
                                                        dash_table.DataTable(
                                                            id="markov-top-items-table",
                                                            columns=[
                                                                {"name": "Rank", "id": "rank"},
                                                                {"name": "Gene / item", "id": "label"},
                                                                {"name": "Priority score", "id": "priority_score"},
                                                                {"name": "Raw diffusion", "id": "raw_diffusion_score"},
                                                                {"name": "Degree", "id": "degree"},
                                                            ],
                                                            data=[],
                                                            page_size=15,
                                                            sort_action="native",
                                                            style_table={"overflowX": "auto"},
                                                            style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"},
                                                        ),
                                                    ],
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="How to Interpret",
                                            value="interpret-tab",
                                            children=[
                                                html.Div(
                                                    style={"padding": "12px 4px", "fontSize": "0.92rem", "color": "#4b5563"},
                                                    children=[
                                                        html.B("Interpretation note: "),
                                                        "This is a network-prioritization score, not a new p-value. Balanced mode combines statistical "
                                                        "evidence with node connectivity so single strong edges do not automatically dominate. The "
                                                        "Top Candidates table is meant to be the first-pass follow-up list. The Diffusion Rankings tab "
                                                        "shows the underlying pathway/gene random-walk priorities.",
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),

                # --------------------------
                # Projection tab
                # --------------------------
                dcc.Tab(
                    label="Projection",
                    value="projection",
                    children=[
                        html.Div(
                            style={"maxWidth": "1200px", "padding": "10px 6px"},
                            children=[
                                html.H3("Pathway ↔ Pathway Projection", style={"color": "#003566"}),
                                html.P(
                                    "This builds a separate pathway-only graph from the current gene ↔ pathway network. "
                                    "Two pathways are connected when they share one or more genes/items. The Network sub-tab shows "
                                    "overlap structure; the Projection Diffusion sub-tab ranks central biological themes inside this pathway-only graph.",
                                    style={"color": "#111827"},
                                ),
                                dcc.Tabs(
                                    id="projection-subtabs",
                                    value="projection-network-tab",
                                    colors={"border": "#CBD5E1", "primary": "#0077b6", "background": "#F8FAFC"},
                                    style={"marginTop": "16px"},
                                    children=[
                                        dcc.Tab(
                                            label="Projection Network",
                                            value="projection-network-tab",
                                            children=[html.Div(style={"padding": "12px 4px"}, children=[
                                                html.Div(
                                                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "10px", "marginTop": "12px"},
                                                    children=[
                                                        html.Div(children=[html.Label("Projection edge weight"), dcc.Dropdown(id="projection-method", options=[
                                                            {"label": "Jaccard similarity — shared / union genes", "value": "jaccard"},
                                                            {"label": "Shared gene count", "value": "shared_count"},
                                                            {"label": "Weighted shared support", "value": "weighted_shared"},
                                                        ], value="jaccard", clearable=False)]),
                                                        html.Div(children=[html.Label("Rows in ranking table"), dcc.Slider(id="projection-top-n", min=10, max=100, step=10, value=30, marks={10: "10", 50: "50", 100: "100"}, tooltip={"placement": "bottom"}, updatemode="mouseup")]),
                                                        html.Div(children=[html.Label("Graph labels"), dcc.Checklist(id="projection-show-labels", options=[{"label": " Show pathway labels", "value": "labels"}], value=["labels"])]),
                                                    ],
                                                ),
                                                html.Button("Build Projection Network", id="btn-build-projection", n_clicks=0, style={"marginTop": "14px", "padding": "10px 14px", "borderRadius": "10px", "border": "none", "background": "#003566", "color": "white", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Button("Download projected_nodes.csv", id="btn-dl-projection-nodes", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Button("Download projected_edges.csv", id="btn-dl-projection-edges", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Button("Download projection graph as SVG", id="btn-dl-projection-svg", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Button("Download projection graph as PDF", id="btn-dl-projection-pdf", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Div("PNG is still available from the Plotly camera icon.", style={"marginTop": "8px", "color": "#6b7280", "fontSize": "0.9rem"}),
                                                html.Div(id="projection-status", style={"marginTop": "10px", "color": "#4b5563"}),
                                                dcc.Graph(id="projection-graph", style={"height": "72vh", "marginTop": "10px"}),
                                                html.H4("Top pathways in projection network", style={"marginTop": "16px", "color": "#003566"}),
                                                dash_table.DataTable(id="projection-ranking-table", columns=[
                                                    {"name": "Rank", "id": "rank"}, {"name": "Pathway / term", "id": "pathway"}, {"name": "Projection score", "id": "projection_score"}, {"name": "Projection degree", "id": "projection_degree"}, {"name": "Original membership degree", "id": "source_degree"},
                                                ], data=[], page_size=15, sort_action="native", style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"}),
                                                html.Div(style={"marginTop": "12px", "fontSize": "0.92rem", "color": "#4b5563"}, children=[html.B("Interpretation note: "), "The projection network is a second graph, not a replacement for the main gene ↔ pathway network. It is best for finding pathway overlap, broad biological themes, and pathway-level hubs."]),
                                            ])],
                                        ),
                                        dcc.Tab(
                                            label="Projection Diffusion",
                                            value="projection-diffusion-tab",
                                            children=[html.Div(style={"padding": "12px 4px"}, children=[
                                                html.H4("Signal diffusion on pathway projection graph", style={"marginTop": "4px", "color": "#003566"}),
                                                html.P("Run random-walk ranking on the pathway-only projection graph. This ranks central biological themes based on pathway overlap, rather than ranking genes and pathways together in the bipartite graph.", style={"color": "#111827"}),
                                                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "10px", "marginTop": "12px"}, children=[
                                                    html.Div(children=[html.Label("Optional projection seed pathway"), dcc.Dropdown(id="projection-diffusion-seed-node", options=[], placeholder="Global projection ranking — no seed selected", clearable=True)]),
                                                    html.Div(children=[html.Label("Walk retention / alpha"), dcc.Slider(id="projection-diffusion-alpha", min=0.50, max=0.95, step=0.05, value=0.85, marks={0.50: "0.50", 0.85: "0.85", 0.95: "0.95"}, tooltip={"placement": "bottom"}, updatemode="mouseup")]),
                                                    html.Div(children=[html.Label("Ranking mode"), dcc.Dropdown(id="projection-diffusion-ranking-mode", options=[
                                                        {"label": "Balanced — overlap + connectivity", "value": "balanced"},
                                                        {"label": "Overlap-weighted — stronger pathway overlaps", "value": "evidence"},
                                                        {"label": "Connectivity-weighted — pathway hubs", "value": "connectivity"},
                                                    ], value="balanced", clearable=False)]),
                                                    html.Div(children=[html.Label("Rows per table"), dcc.Slider(id="projection-diffusion-top-n", min=10, max=100, step=10, value=30, marks={10: "10", 50: "50", 100: "100"}, tooltip={"placement": "bottom"}, updatemode="mouseup")]),
                                                ]),
                                                html.Button("Run Projection Diffusion Analysis", id="btn-run-projection-diffusion", n_clicks=0, style={"marginTop": "14px", "padding": "10px 14px", "borderRadius": "10px", "border": "none", "background": "#003566", "color": "white", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Button("Download projection_diffusion_results.csv", id="btn-dl-projection-diffusion", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Button("Download projection_top_candidates.csv", id="btn-dl-projection-candidates", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                                html.Div(id="projection-diffusion-status", style={"marginTop": "10px", "color": "#4b5563"}),
                                                html.Div(id="projection-diffusion-summary-cards", style={"marginTop": "12px"}),
                                                dcc.Tabs(id="projection-diffusion-result-subtabs", value="projection-candidates-tab", colors={"border": "#CBD5E1", "primary": "#0077b6", "background": "#F8FAFC"}, style={"marginTop": "16px"}, children=[
                                                    dcc.Tab(label="Top Candidates", value="projection-candidates-tab", children=[html.Div(style={"padding": "12px 4px"}, children=[
                                                        html.H4("Top follow-up candidates — projection pathways", style={"marginTop": "4px", "color": "#003566"}),
                                                        dash_table.DataTable(id="projection-diffusion-candidates-table", columns=[
                                                            {"name": "Rank", "id": "candidate_rank"}, {"name": "Pathway / term", "id": "label"}, {"name": "Follow-up score", "id": "followup_score"}, {"name": "Diffusion priority", "id": "priority_score"}, {"name": "Projection degree", "id": "degree"}, {"name": "Original membership degree", "id": "source_degree"}, {"name": "Mean overlap weight", "id": "mean_overlap_weight"}, {"name": "Max shared items", "id": "max_shared_count"},
                                                        ], data=[], page_size=15, sort_action="native", style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"}),
                                                    ])]),
                                                    dcc.Tab(label="Diffusion Rankings", value="projection-rankings-tab", children=[html.Div(style={"padding": "12px 4px"}, children=[
                                                        html.H4("Top pathways by projection diffusion score", style={"marginTop": "4px", "color": "#003566"}),
                                                        dash_table.DataTable(id="projection-diffusion-rankings-table", columns=[
                                                            {"name": "Rank", "id": "rank"}, {"name": "Pathway / term", "id": "label"}, {"name": "Priority score", "id": "priority_score"}, {"name": "Raw diffusion score", "id": "raw_diffusion_score"}, {"name": "Projection degree", "id": "degree"}, {"name": "Original membership degree", "id": "source_degree"}, {"name": "Ranking mode", "id": "ranking_mode"},
                                                        ], data=[], page_size=15, sort_action="native", style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"}),
                                                    ])]),
                                                ]),
                                                html.Div(style={"marginTop": "12px", "fontSize": "0.92rem", "color": "#4b5563"}, children=[html.B("Interpretation note: "), "Projection diffusion ranks pathway-level themes based on overlap structure. It is complementary to the main bipartite diffusion tab, not a replacement."]),
                                            ])],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                # --------------------------
                # Consensus tab
                # --------------------------
                dcc.Tab(
                    label="Consensus",
                    value="consensus",
                    children=[
                        html.Div(
                            style={"maxWidth": "1200px", "padding": "10px 6px"},
                            children=[
                                html.H3("Consensus Follow-up Candidates", style={"color": "#003566"}),
                                html.P(
                                    "This combines the main bipartite diffusion candidates with the pathway-projection diffusion candidates. "
                                    "It highlights pathways that are both directly supported by the enrichment network and central within broader pathway-overlap biology.",
                                    style={"color": "#111827"},
                                ),
                                html.Div(
                                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "10px", "marginTop": "12px"},
                                    children=[
                                        html.Div(children=[
                                            html.Label("Rows in consensus table"),
                                            dcc.Slider(id="consensus-top-n", min=10, max=100, step=10, value=30, marks={10: "10", 50: "50", 100: "100"}, tooltip={"placement": "bottom"}, updatemode="mouseup"),
                                        ]),
                                    ],
                                ),
                                html.Button("Build Consensus Candidates", id="btn-build-consensus", n_clicks=0, style={"marginTop": "14px", "padding": "10px 14px", "borderRadius": "10px", "border": "none", "background": "#003566", "color": "white", "fontWeight": "800", "cursor": "pointer"}),
                                html.Button("Download consensus_candidates.csv", id="btn-dl-consensus", n_clicks=0, style={"marginTop": "14px", "marginLeft": "8px", "padding": "10px 14px", "borderRadius": "10px", "border": "1px solid #003566", "background": "white", "color": "#003566", "fontWeight": "800", "cursor": "pointer"}),
                                html.Div(id="consensus-status", style={"marginTop": "10px", "color": "#4b5563"}),
                                html.Div(id="consensus-summary-cards", style={"marginTop": "12px"}),
                                html.H4("Top consensus candidates", style={"marginTop": "16px", "color": "#003566"}),
                                dash_table.DataTable(
                                    id="consensus-candidates-table",
                                    columns=[
                                        {"name": "Consensus rank", "id": "consensus_rank"},
                                        {"name": "Pathway / term", "id": "label"},
                                        {"name": "Consensus score", "id": "consensus_score"},
                                        {"name": "Bipartite follow-up score", "id": "bipartite_followup_score"},
                                        {"name": "Projection follow-up score", "id": "projection_followup_score"},
                                        {"name": "Bipartite rank", "id": "bipartite_candidate_rank"},
                                        {"name": "Projection rank", "id": "projection_candidate_rank"},
                                        {"name": "Best adj p", "id": "best_adj_p"},
                                        {"name": "Max evidence", "id": "max_evidence"},
                                        {"name": "Bipartite degree", "id": "bipartite_degree"},
                                        {"name": "Projection degree", "id": "projection_degree"},
                                        {"name": "Mean overlap", "id": "mean_overlap_weight"},
                                        {"name": "Max shared items", "id": "max_shared_count"},
                                    ],
                                    data=[], page_size=20, sort_action="native", style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Poppins, sans-serif", "padding": "8px", "textAlign": "left"},
                                ),
                                html.Div(style={"marginTop": "12px", "fontSize": "0.92rem", "color": "#4b5563"}, children=[html.B("Interpretation note: "), "Consensus score is not a p-value. It is a pragmatic final prioritization score that rewards pathways supported by both the direct bipartite network and the pathway-overlap projection network."]),
                            ],
                        )
                    ],
                ),

                # --------------------------
                # README tab
                # --------------------------
                dcc.Tab(
                    label="README",
                    value="readme",
                    children=[
                        html.Div(
                            style={"maxWidth": "900px", "padding": "10px 6px"},
                            children=[
                                html.H3("How to read this network", style={"color": "#003566"}),

                                html.P(
                                    "This visualization shows relationships between genes (items) and enriched terms/pathways (groups).",
                                    style={"color": "#111827"},
                                ),

                                html.H4("Nodes", style={"marginTop": "12px", "color": "#003566"}),
                                html.Ul(
                                    [
                                        html.Li("Genes are item nodes."),
                                        html.Li("Terms/pathways are group nodes."),
                                        html.Li("Node size reflects connectivity (degree), not biological importance."),
                                    ],
                                    style={"color": "#111827"},
                                ),

                                html.H4("Edges", style={"marginTop": "12px", "color": "#003566"}),
                                html.Ul(
                                    [
                                        html.Li("An edge indicates a gene belongs to / maps to a term."),
                                        html.Li(
                                            [
                                                html.B("Edge color: "),
                                                "edges connected to the same term share the same color (helps trace term membership).",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.B("Edge thickness (optional): "),
                                                "when “Thickness by weight” is enabled, thicker edges indicate stronger statistical evidence "
                                                "(higher −log10 adjusted p-value).",
                                            ]
                                        ),
                                    ],
                                    style={"color": "#111827"},
                                ),

                                html.H4("Interpretation notes", style={"marginTop": "12px", "color": "#003566"}),
                                html.Ul(
                                    [
                                        html.Li(
                                            "Thickness is scaled for visual contrast (relative within the current view), "
                                            "not meant as a precise numeric axis."
                                        ),
                                        html.Li(
                                            "Use this plot for pattern discovery; use the tables/CSV exports for exact p-values."
                                        ),
                                    ],
                                    style={"color": "#111827"},
                                ),

                                html.H4("Controls", style={"marginTop": "12px", "color": "#003566"}),
                                html.Ul(
                                    [
                                        html.Li("Search filters nodes shown in the plot."),
                                        html.Li("Minimum node degree hides weakly connected nodes."),
                                        html.Li("Edge width range controls the min/max visible thickness (when enabled)."),
                                        html.Li("Weight range controls how weights map into thickness."),
                                    ],
                                    style={"color": "#111827"},
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),

        # --------------------------
        # Stores
        # --------------------------
        dcc.Store(id="store-raw-df"),
        dcc.Store(id="store-graph"),           # serialized nodes/edges + attrs
        dcc.Store(id="store-nodes-edges-df"),  # for downloads
        dcc.Store(id="store-markov-results"),  # signal diffusion / Markov results
        dcc.Download(id="dl-markov"),
        dcc.Download(id="dl-candidates"),
        dcc.Store(id="store-projection-graph"),
        dcc.Store(id="store-projection-export"),
        dcc.Store(id="store-projection-diffusion-results"),
        dcc.Download(id="dl-projection-nodes"),
        dcc.Download(id="dl-projection-edges"),
        dcc.Download(id="dl-projection-diffusion"),
        dcc.Download(id="dl-projection-candidates"),
        dcc.Store(id="store-consensus-results"),
        dcc.Download(id="dl-consensus"),
        dcc.Download(id="dl-main-svg"),
        dcc.Download(id="dl-main-pdf"),
        dcc.Download(id="dl-projection-svg"),
        dcc.Download(id="dl-projection-pdf"),
        dcc.Download(id="dl-report-bundle"),
        dcc.Download(id="dl-llm-triage-bundle"),
    ],
)


# ----------------------------
# Layout
# ----------------------------
app.layout = html.Div(
    style={
        "display": "flex",
        "gap": "16px",
        "padding": "16px",
        "fontFamily": "Poppins, sans-serif",
        "background": "#f1f5f9",
        "minHeight": "100vh",
    },
    children=[
        # ==========================
        # SIDEBAR (uploads + filters)
        # ==========================
        html.Div(
            style={
                "width": "360px",
                "background": "#EEF2FF",  # was: ""#EEF2FF"  (invalid)
                "borderRadius": "14px",
                "padding": "14px",
                "boxShadow": "0 4px 10px rgba(0,0,0,0.12)",
                "height": "calc(100vh - 32px)",
                "overflow": "auto",
            },
            children=[
                html.H3(
                    "JC Enrichment Network Studio",
                    style={"marginBottom": "6px", "color": "#003566"},
                ),

                html.Div(
                    REQUIRED_RAW_COLS_NOTE,
                    style={
                        "fontSize": "0.9rem",
                        "color": "#4b5563",
                        "marginBottom": "12px",
                    },
                ),

                # -------- Raw input --------
                html.H4(
                    "Raw input",
                    style={"marginTop": "10px", "color": "#003566"},
                ),

                dcc.Upload(
                    id="upload-raw",
                    children=html.Div(
                        ["Drag & drop or ", html.B("browse"), " (CSV)"]
                    ),
                    style={
                        "width": "100%",
                        "padding": "14px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "12px",
                        "textAlign": "center",
                        "cursor": "pointer",
                        "color": "#111827",
                        "background": "#f8fafc",
                    },
                    multiple=False,
                ),

                html.Div(id="raw-upload-status", style={"marginTop": "8px", "fontSize": "0.9rem"}),
                html.Button(
                    "Load demo dataset",
                    id="btn-load-demo",
                    n_clicks=0,
                    style={
                        "marginTop": "10px",
                        "width": "100%",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "1px solid #003566",
                        "background": "white",
                        "color": "#003566",
                        "fontWeight": "800",
                        "cursor": "pointer",
                    },
                ),

                html.Div(
                    id="demo-status",
                    style={
                        "marginTop": "8px",
                        "fontSize": "0.9rem",
                        "color": "#065f46",
                    },
                ),
                html.Div(
                    style={"marginTop": "12px"},
                    children=[
                        html.Label("Column mapping preset"),
                        dcc.Dropdown(
                            id="column-preset",
                            options=COLUMN_MAPPING_PRESETS,
                            value="custom_long",
                            clearable=False,
                        ),
                        html.Button(
                            "Apply column preset",
                            id="btn-apply-preset",
                            n_clicks=0,
                            style={
                                "marginTop": "8px",
                                "width": "100%",
                                "padding": "9px",
                                "borderRadius": "10px",
                                "border": "1px solid #003566",
                                "background": "white",
                                "color": "#003566",
                                "fontWeight": "800",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(
                            id="preset-status",
                            style={
                                "marginTop": "8px",
                                "fontSize": "0.9rem",
                                "color": "#475569",
                            },
                        ),

                        html.Label("Item column (genes)", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="item-col", placeholder="Select…"),

                        html.Label("Group column (pathways/terms)", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="group-col", placeholder="Select…"),

                        html.Label("Weight/score column (optional)", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="weight-col", placeholder="None", clearable=True),
                    ],
                ),

                html.Button(
                    "Preprocess / Build graph",
                    id="btn-build",
                    n_clicks=0,
                    style={
                        "marginTop": "12px",
                        "width": "100%",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "none",
                        "background": "#0077b6",
                        "color": "white",
                        "fontWeight": "800",
                        "cursor": "pointer",
                    },
                ),

                html.Div(id="graph-size-warning"),

                html.Div(
                    style={
                        "border": "1px solid #CBD5E1",
                        "borderRadius": "12px",
                        "padding": "12px",
                        "background": "#F8FAFC",
                        "marginTop": "12px",
                    },
                    children=[
                        html.H4(
                            "Final report bundle",
                            style={"marginTop": 0, "marginBottom": "6px", "color": "#003566"},
                        ),
                        html.Div(
                            "Use this after building the graph, projection, diffusion analyses, and consensus candidates.",
                            style={"fontSize": "0.86rem", "color": "#475569", "marginBottom": "10px"},
                        ),
                        html.Button(
                            "Download full report bundle",
                            id="btn-dl-report-bundle",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "borderRadius": "10px",
                                "border": "none",
                                "background": "#003566",
                                "color": "white",
                                "fontWeight": "800",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(
                            id="report-bundle-status",
                            style={
                                "marginTop": "8px",
                                "fontSize": "0.86rem",
                                "color": "#475569",
                                "lineHeight": "1.35",
                            },
                        ),
                        html.Div(
                            id="report-bundle-running-status",
                            style={
                                "marginTop": "6px",
                                "fontSize": "0.86rem",
                                "color": "#7c2d12",
                                "lineHeight": "1.35",
                            },
                        ),
                    ],
                ),

                html.Div(
                    style={
                        "border": "1px solid #CBD5E1",
                        "borderRadius": "12px",
                        "padding": "12px",
                        "background": "#F8FAFC",
                        "marginTop": "12px",
                    },
                    children=[
                        html.H4(
                            "LLM Triage export",
                            style={"marginTop": 0, "marginBottom": "6px", "color": "#003566"},
                        ),
                        html.Div(
                            "Use this after running the graph, projection, diffusion analyses, and consensus candidates. "
                            "This creates an input bundle for the companion LLM Triage workflow. It does not run an LLM or spend API credits.",
                            style={"fontSize": "0.86rem", "color": "#475569", "marginBottom": "10px"},
                        ),
                        html.Button(
                            "Export for LLM Triage",
                            id="btn-dl-llm-triage-bundle",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "borderRadius": "10px",
                                "border": "1px solid #003566",
                                "background": "white",
                                "color": "#003566",
                                "fontWeight": "800",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(
                            id="llm-triage-bundle-status",
                            style={
                                "marginTop": "8px",
                                "fontSize": "0.86rem",
                                "color": "#475569",
                                "lineHeight": "1.35",
                            },
                        ),
                        html.Div(
                            id="llm-triage-bundle-running-status",
                            style={
                                "marginTop": "6px",
                                "fontSize": "0.86rem",
                                "color": "#7c2d12",
                                "lineHeight": "1.35",
                            },
                        ),
                    ],
                ),

                # -------- Sidebar controls (Plot controls + Filters) --------
                html.Hr(style={"margin": "14px 0"}),

                html.H4("Plot controls", style={"color": "#003566"}),

                html.Label("Search gene/pathway"),
                dcc.Input(
                    id="search",
                    placeholder="Type to filter…",
                    debounce=True,
                    style={
                        "padding": "10px 12px",
                        "borderRadius": "10px",
                        "border": "1px solid #d1d5db",
                        "width": "100%",
                        "marginTop": "6px",
                    },
                ),

                html.Label("Layout", style={"marginTop": "10px"}),
                dcc.Dropdown(
                    id="layout-mode",
                    options=[
                        {"label": "Bipartite (two columns)", "value": "bipartite"},
                        {"label": "Force-directed", "value": "force"},
                    ],
                    value="bipartite",
                    clearable=False,
                ),

                dcc.Checklist(
                    id="edge-style",
                    options=[{"label": " Thickness by weight", "value": "thick"}],
                    value=["thick"],
                    style={"marginTop": "10px"},
                ),

                dcc.Checklist(
                    id="candidate-highlight-toggle",
                    options=[{"label": " Highlight top follow-up candidates", "value": "highlight"}],
                    value=["highlight"],
                    style={"marginTop": "10px"},
                ),

                html.Label("Number of candidates to highlight", style={"marginTop": "10px"}),
                dcc.Slider(
                    id="candidate-highlight-count",
                    min=3,
                    max=25,
                    step=1,
                    value=10,
                    marks={3: "3", 10: "10", 25: "25"},
                    tooltip={"placement": "right"},
                    updatemode="mouseup",
                ),

                html.Hr(style={"margin": "14px 0"}),

                html.H4("Filters", style={"color": "#003566"}),

                html.Label("Minimum node degree"),
                dcc.Slider(
                    id="min-degree",
                    min=0, max=20, step=1, value=0,
                    marks=None,
                    tooltip={"placement": "right"},
                    updatemode="mouseup",
                ),

                html.Label("Edge width range", style={"marginTop": "10px"}),
                dcc.RangeSlider(
                    id="edge-width-range",
                    min=1,
                    max=10,
                    step=0.5,
                    value=[1.5, 6.0],
                    marks=None,
                    tooltip={"placement": "right"},
                    updatemode="mouseup",
                ),

                html.Label("Weight range for scaling (−log10 adj p)", style={"marginTop": "10px"}),
                dcc.RangeSlider(
                    id="edge-weight-range",
                    min=0,
                    max=20,
                    step=0.5,
                    value=[0, 10],
                    marks=None,
                    tooltip={"placement": "right"},
                    updatemode="mouseup",
                ),

                html.Label("Minimum edge weight", style={"marginTop": "10px"}),
                dcc.Slider(
                    id="min-weight",
                    min=0, max=10, step=0.5, value=0,
                    marks=None,
                    tooltip={"placement": "right"},
                    updatemode="mouseup",
                ),

                html.Label("Maximum number of groups", style={"marginTop": "10px"}),
                dcc.Slider(
                    id="max-groups",
                    min=5, max=200, step=5, value=50,
                    marks=None,
                    tooltip={"placement": "right"},
                    updatemode="mouseup",
                ),

                # -------- Advanced uploads (optional) --------
                html.Hr(style={"margin": "14px 0"}),

                html.H4("Advanced (optional)", style={"color": "#003566"}),
                dcc.Checklist(
                    id="use-prebuilt",
                    options=[{"label": " Use prebuilt nodes/edges instead", "value": "yes"}],
                    value=[],
                    style={"marginBottom": "8px"},
                ),

                dcc.Upload(
                    id="upload-nodes",
                    children=html.Div(["Upload nodes.csv"]),
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "background": "#f8fafc",
                    },
                    multiple=False,
                ),

                dcc.Upload(
                    id="upload-edges",
                    children=html.Div(["Upload edges.csv"]),
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "background": "#f8fafc",
                        "marginTop": "8px",
                    },
                    multiple=False,
                ),

                html.Div(id="prebuilt-status", style={"marginTop": "8px", "fontSize": "0.9rem"}),

                # -------- Export --------
                html.Hr(style={"margin": "14px 0"}),

                html.H4("Export", style={"color": "#003566"}),
                html.Div("Downloads appear after graph build.", style={"fontSize": "0.9rem", "color": "#4b5563"}),

                dcc.Download(id="dl-nodes"),
                dcc.Download(id="dl-edges"),

                html.Button(
                    "Download nodes.csv",
                    id="btn-dl-nodes",
                    n_clicks=0,
                    style={
                        "marginTop": "10px",
                        "width": "100%",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "1px solid #0077b6",
                        "background": "white",
                        "color": "#0077b6",
                        "fontWeight": "800",
                        "cursor": "pointer",
                    },
                ),

                html.Button(
                    "Download edges.csv",
                    id="btn-dl-edges",
                    n_clicks=0,
                    style={
                        "marginTop": "8px",
                        "width": "100%",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "1px solid #0077b6",
                        "background": "white",
                        "color": "#0077b6",
                        "fontWeight": "800",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        main_panel,
    ],
)



# ----------------------------
# Figure export helpers
# ----------------------------

def build_main_graph_figure_for_export(
    graph_data,
    search,
    min_degree,
    min_weight,
    max_groups,
    layout_mode,
    markov_results,
    candidate_highlight_toggle,
    candidate_highlight_count,
) -> go.Figure:
    """Rebuild the currently displayed main bipartite graph figure for SVG/PDF export."""
    if not graph_data:
        return go.Figure()

    g = rebuild_graph_from_store(graph_data)
    sg = subgraph_filter(
        g,
        search=search or "",
        min_degree=int(min_degree or 0),
        min_weight=float(min_weight or 0),
        max_groups=int(max_groups or 50),
        largest_component_only=False,
    )

    if (
        sg.number_of_nodes() > VERY_LARGE_RENDER_NODE_LIMIT
        or sg.number_of_edges() > VERY_LARGE_RENDER_EDGE_LIMIT
    ):
        st = graph_stats(sg)
        fig = large_graph_placeholder_figure(sg.number_of_nodes(), sg.number_of_edges())
        cards = html.Div(
            style={
                "border": "1px solid #fed7aa",
                "borderLeft": "5px solid #f97316",
                "borderRadius": "12px",
                "padding": "12px",
                "background": "#fff7ed",
                "color": "#7c2d12",
            },
            children=[
                html.Div("Filtered graph is too large to render smoothly.", style={"fontWeight": "800"}),
                html.Div(
                    f"Filtered graph: {st['nodes']:,} nodes, {st['edges']:,} edges. "
                    "Increase min degree/edge weight, reduce max groups, or search for a pathway family."
                ),
            ],
        )
        top_groups_data = [{"group": name, "degree": deg} for name, deg in st["top_groups"]]
        top_items_data = [{"item": name, "degree": deg} for name, deg in st["top_items"]]
        return fig, cards, top_groups_data, top_items_data

    if layout_mode == "force":
        pos = nx.spring_layout(sg, seed=7, k=1 / math.sqrt(max(1, sg.number_of_nodes())))
    else:
        pos = layout_bipartite_two_column(sg)

    highlight_nodes = {}
    if "highlight" in (candidate_highlight_toggle or []) and markov_results:
        for row in (markov_results.get("candidate_rows") or [])[: int(candidate_highlight_count or 10)]:
            node_id = row.get("node_id")
            if node_id in sg.nodes:
                highlight_nodes[node_id] = row

    fig = make_plotly_network(sg, pos, show_labels=False, highlight_nodes=highlight_nodes)
    fig.update_layout(width=1400, height=1000)
    return fig


def _download_plotly_figure(fig: go.Figure, file_format: str, filename: str):
    """Return a Dash download payload for Plotly static image export via Kaleido."""
    image_bytes = fig.to_image(format=file_format, width=1400, height=1000, scale=1)
    return dcc.send_bytes(lambda buffer: buffer.write(image_bytes), filename)


# ----------------------------
# Full report bundle helpers
# ----------------------------

def _safe_csv_from_store(store: dict | None, key: str) -> str:
    """Safely pull a CSV string from a Dash store."""
    if not store:
        return ""
    return store.get(key, "") or ""


def _graph_stats_csv_from_graph_store(graph_data: dict | None) -> str:
    """Build a compact graph_stats.csv from the current main graph."""
    if not graph_data:
        return ""

    g = rebuild_graph_from_store(graph_data)
    st = graph_stats(g)

    rows = [
        {"metric": "nodes", "value": st.get("nodes", 0)},
        {"metric": "edges", "value": st.get("edges", 0)},
        {"metric": "items", "value": st.get("items", 0)},
        {"metric": "groups", "value": st.get("groups", 0)},
        {"metric": "components", "value": st.get("components", 0)},
        {"metric": "largest_component", "value": st.get("largest_component", 0)},
        {"metric": "weight_key", "value": st.get("weight_key", "")},
    ]

    weight_summary = st.get("weight_summary") or {}
    for key, value in weight_summary.items():
        rows.append({"metric": f"weight_{key}", "value": value})

    return pd.DataFrame(rows).to_csv(index=False)


def _run_summary_html(
    graph_data: dict | None,
    projection_data: dict | None,
    markov_store: dict | None,
    projection_diffusion_store: dict | None,
    consensus_store: dict | None,
    mapped_columns: dict,
    settings: dict,
) -> str:
    """Human-readable HTML report summary for the zip bundle."""
    g = rebuild_graph_from_store(graph_data) if graph_data else nx.Graph()
    pg = rebuild_projection_from_store(projection_data) if projection_data else nx.Graph()

    main_stats = graph_stats(g) if graph_data else {
        "nodes": 0,
        "edges": 0,
        "items": 0,
        "groups": 0,
        "components": 0,
        "largest_component": 0,
    }

    top_bipartite = ""
    if markov_store and markov_store.get("candidate_rows"):
        top_bipartite = markov_store["candidate_rows"][0].get("label", "")

    top_projection = ""
    if projection_diffusion_store and projection_diffusion_store.get("candidate_rows"):
        top_projection = projection_diffusion_store["candidate_rows"][0].get("label", "")

    top_consensus = ""
    if consensus_store and consensus_store.get("candidate_rows"):
        top_consensus = consensus_store["candidate_rows"][0].get("label", "")

    created_at = datetime.now(timezone.utc).isoformat()

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Enrichment Network Studio Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 980px; margin: 32px auto; line-height: 1.45; color: #111827; }}
    h1, h2 {{ color: #003566; }}
    code, pre {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px; margin: 12px 0; background: #f8fafc; }}
    .warning {{ border-left: 5px solid #f97316; padding: 10px 14px; background: #fff7ed; }}
  </style>
</head>
<body>
  <h1>Enrichment Network Studio Report</h1>
  <p><strong>Created:</strong> {created_at}</p>

  <div class="warning">
    <strong>Important:</strong> Network diffusion, follow-up, and consensus scores are prioritization scores,
    not statistical p-values. Use adjusted p-values from the original enrichment input as statistical evidence.
  </div>

  <h2>Mapped Columns</h2>
  <pre>{json.dumps(mapped_columns, indent=2)}</pre>

  <h2>Run Settings</h2>
  <pre>{json.dumps(settings, indent=2)}</pre>

  <h2>Main Graph Summary</h2>
  <div class="card">
    <p><strong>Nodes:</strong> {main_stats.get("nodes", 0)}</p>
    <p><strong>Edges:</strong> {main_stats.get("edges", 0)}</p>
    <p><strong>Items/genes:</strong> {main_stats.get("items", 0)}</p>
    <p><strong>Groups/pathways:</strong> {main_stats.get("groups", 0)}</p>
    <p><strong>Components:</strong> {main_stats.get("components", 0)}</p>
    <p><strong>Largest component:</strong> {main_stats.get("largest_component", 0)}</p>
  </div>

  <h2>Projection Summary</h2>
  <div class="card">
    <p><strong>Projection nodes:</strong> {pg.number_of_nodes()}</p>
    <p><strong>Projection edges:</strong> {pg.number_of_edges()}</p>
  </div>

  <h2>Top Results</h2>
  <div class="card">
    <p><strong>Top bipartite follow-up candidate:</strong> {top_bipartite or "Not run"}</p>
    <p><strong>Top projection follow-up candidate:</strong> {top_projection or "Not run"}</p>
    <p><strong>Top consensus candidate:</strong> {top_consensus or "Not built"}</p>
  </div>

  <h2>How to Read This Bundle</h2>
  <ul>
    <li><strong>main_nodes.csv / main_edges.csv:</strong> reproducible main gene-pathway graph tables.</li>
    <li><strong>projection_nodes.csv / projection_edges.csv:</strong> pathway-overlap graph tables.</li>
    <li><strong>bipartite_diffusion_results.csv:</strong> random-walk ranking on the main graph.</li>
    <li><strong>bipartite_top_candidates.csv:</strong> practical follow-up candidates from main graph diffusion.</li>
    <li><strong>projection_diffusion_results.csv:</strong> random-walk ranking on the pathway projection graph.</li>
    <li><strong>projection_top_candidates.csv:</strong> practical follow-up candidates from projection diffusion.</li>
    <li><strong>consensus_candidates.csv:</strong> agreement between main graph and projection candidate rankings.</li>
  </ul>
</body>
</html>
"""


def _write_optional_graph_images_to_zip(
    z: zipfile.ZipFile,
    graph_data: dict | None,
    projection_data: dict | None,
    search,
    min_degree,
    min_weight,
    max_groups,
    layout_mode,
    markov_results,
    candidate_highlight_toggle,
    candidate_highlight_count,
    projection_show_labels,
) -> None:
    """
    Try to include SVG/PDF graph images.
    If Kaleido/static export fails, write a note instead of crashing the bundle.
    """
    try:
        if graph_data:
            main_fig = build_main_graph_figure_for_export(
                graph_data,
                search,
                min_degree,
                min_weight,
                max_groups,
                layout_mode,
                markov_results,
                candidate_highlight_toggle,
                candidate_highlight_count,
            )
            z.writestr("report_bundle/main_graph.svg", main_fig.to_image(format="svg", width=1400, height=1000, scale=1))
            z.writestr("report_bundle/main_graph.pdf", main_fig.to_image(format="pdf", width=1400, height=1000, scale=1))
    except Exception as e:
        z.writestr(
            "report_bundle/main_graph_export_note.txt",
            f"Main graph image export failed. CSV tables are still included. Error: {e}",
        )

    try:
        if projection_data:
            pg = rebuild_projection_from_store(projection_data)
            show_labels = "labels" in (projection_show_labels or [])
            projection_fig = make_projection_figure(pg, show_labels=show_labels)
            projection_fig.update_layout(width=1400, height=1000)
            z.writestr("report_bundle/projection_graph.svg", projection_fig.to_image(format="svg", width=1400, height=1000, scale=1))
            z.writestr("report_bundle/projection_graph.pdf", projection_fig.to_image(format="pdf", width=1400, height=1000, scale=1))
    except Exception as e:
        z.writestr(
            "report_bundle/projection_graph_export_note.txt",
            f"Projection graph image export failed. CSV tables are still included. Error: {e}",
        )


def build_report_bundle_zip(
    raw_json,
    graph_data,
    nodes_edges_store,
    projection_data,
    projection_export_store,
    markov_store,
    projection_diffusion_store,
    consensus_store,
    mapped_columns,
    settings,
    search=None,
    min_degree=0,
    min_weight=0,
    max_groups=50,
    layout_mode="bipartite",
    candidate_highlight_toggle=None,
    candidate_highlight_count=10,
    projection_show_labels=None,
) -> bytes:
    """Build the full Network Studio report bundle zip."""
    buf = io.BytesIO()

    run_summary = _run_summary_html(
        graph_data=graph_data,
        projection_data=projection_data,
        markov_store=markov_store,
        projection_diffusion_store=projection_diffusion_store,
        consensus_store=consensus_store,
        mapped_columns=mapped_columns,
        settings=settings,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product": "Enrichment Network Studio",
        "bundle_type": "network_studio_report_bundle",
        "mapped_columns": mapped_columns,
        "settings": settings,
        "caveat": "Diffusion, follow-up, and consensus scores are prioritization scores, not statistical p-values.",
    }

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("report_bundle/run_summary.html", run_summary)
        z.writestr("report_bundle/manifest.json", json.dumps(manifest, indent=2))
        z.writestr("report_bundle/mapped_columns.json", json.dumps(mapped_columns, indent=2))
        z.writestr("report_bundle/settings.json", json.dumps(settings, indent=2))

        if raw_json:
            try:
                raw_df = pd.read_json(StringIO(raw_json), orient="split")
                z.writestr("report_bundle/input_preview.csv", raw_df.head(500).to_csv(index=False))
            except Exception as e:
                z.writestr("report_bundle/input_preview_error.txt", f"Could not export input preview: {e}")

        z.writestr("report_bundle/main_nodes.csv", _safe_csv_from_store(nodes_edges_store, "nodes_csv"))
        z.writestr("report_bundle/main_edges.csv", _safe_csv_from_store(nodes_edges_store, "edges_csv"))
        z.writestr("report_bundle/graph_stats.csv", _graph_stats_csv_from_graph_store(graph_data))

        z.writestr("report_bundle/projection_nodes.csv", _safe_csv_from_store(projection_export_store, "nodes_csv"))
        z.writestr("report_bundle/projection_edges.csv", _safe_csv_from_store(projection_export_store, "edges_csv"))

        z.writestr("report_bundle/bipartite_diffusion_results.csv", _safe_csv_from_store(markov_store, "csv"))
        z.writestr("report_bundle/bipartite_top_candidates.csv", _safe_csv_from_store(markov_store, "candidate_csv"))
        z.writestr("report_bundle/projection_diffusion_results.csv", _safe_csv_from_store(projection_diffusion_store, "csv"))
        z.writestr("report_bundle/projection_top_candidates.csv", _safe_csv_from_store(projection_diffusion_store, "candidate_csv"))
        z.writestr("report_bundle/consensus_candidates.csv", _safe_csv_from_store(consensus_store, "csv"))

        z.writestr(
            "report_bundle/interpretation_notes.md",
            """# Interpretation Notes

Adjusted p-value = statistical enrichment evidence from the input table.

Edge weight = transformed evidence, usually -log10(adjusted p-value).

Diffusion score = network prioritization score, not a p-value.

Follow-up score = practical triage score combining diffusion, evidence, and support breadth.

Consensus score = agreement between main bipartite and pathway-projection rankings.

Projection network = pathway-overlap graph, not a replacement for the main gene-pathway graph.
""",
        )

        _write_optional_graph_images_to_zip(
            z=z,
            graph_data=graph_data,
            projection_data=projection_data,
            search=search,
            min_degree=min_degree,
            min_weight=min_weight,
            max_groups=max_groups,
            layout_mode=layout_mode,
            markov_results=markov_store,
            candidate_highlight_toggle=candidate_highlight_toggle,
            candidate_highlight_count=candidate_highlight_count,
            projection_show_labels=projection_show_labels,
        )

    return buf.getvalue()

def build_llm_triage_bundle_zip(
    raw_json,
    graph_data,
    nodes_edges_store,
    projection_data,
    projection_export_store,
    markov_store,
    projection_diffusion_store,
    consensus_store,
    mapped_columns,
    settings,
) -> bytes:
    """
    Build the companion LLM Triage input bundle.

    This export is intentionally API-safe: it does not call an LLM, PubMed,
    OpenAI, or any paid service. It only packages Network Studio outputs into
    a clean folder that a local/BYOK/paid/consulting LLM Triage workflow can ingest.
    """
    buf = io.BytesIO()

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product": "Enrichment Network Studio",
        "bundle_type": "llm_triage_input_bundle",
        "purpose": "Input package for the companion Enrichment LLM Triage workflow.",
        "api_safety": "This export does not run an LLM, call PubMed, or spend API credits.",
        "important_note": (
            "Diffusion, follow-up, and consensus scores are prioritization scores, "
            "not statistical p-values. Use adjusted p-values from the original input "
            "as statistical enrichment evidence."
        ),
    }

    manifest = {
        "created_at": run_summary["created_at"],
        "product": "Enrichment Network Studio",
        "bundle_type": "llm_triage_input_bundle",
        "mapped_columns": mapped_columns,
        "settings": settings,
        "expected_consumer": "Enrichment LLM Triage companion app or paid/BYOK/local workflow",
        "api_safety": "No live LLM call is made by this export.",
    }

    interpretation_notes = """# LLM Triage Input Notes

This bundle was generated by Enrichment Network Studio.

It is meant to be uploaded into, or consumed by, the companion Enrichment LLM Triage workflow.

This public Network Studio export does **not** run an LLM, call PubMed, or spend API credits.

## Score caveats

- Adjusted p-value = statistical enrichment evidence from the original input table.
- Edge weight = transformed evidence, usually -log10(adjusted p-value).
- Diffusion score = network prioritization score, not a p-value.
- Follow-up score = practical triage score combining diffusion, evidence, and support breadth.
- Consensus score = agreement between main bipartite and pathway-projection rankings.
- Projection network = pathway-overlap graph, not a replacement for the main gene-pathway graph.

LLM interpretation is assistive and should be reviewed by a scientist.
"""

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("llm_triage_input/run_summary.json", json.dumps(run_summary, indent=2))
        z.writestr("llm_triage_input/mapped_columns.json", json.dumps(mapped_columns, indent=2))
        z.writestr("llm_triage_input/settings_manifest.json", json.dumps(manifest, indent=2))
        z.writestr("llm_triage_input/interpretation_notes.md", interpretation_notes)

        if raw_json:
            try:
                raw_df = pd.read_json(StringIO(raw_json), orient="split")
                z.writestr("llm_triage_input/input_preview.csv", raw_df.head(500).to_csv(index=False))
            except Exception as e:
                z.writestr("llm_triage_input/input_preview_error.txt", f"Could not export input preview: {e}")

        z.writestr("llm_triage_input/main_nodes.csv", _safe_csv_from_store(nodes_edges_store, "nodes_csv"))
        z.writestr("llm_triage_input/main_edges.csv", _safe_csv_from_store(nodes_edges_store, "edges_csv"))
        z.writestr("llm_triage_input/projection_nodes.csv", _safe_csv_from_store(projection_export_store, "nodes_csv"))
        z.writestr("llm_triage_input/projection_edges.csv", _safe_csv_from_store(projection_export_store, "edges_csv"))
        z.writestr("llm_triage_input/bipartite_diffusion_results.csv", _safe_csv_from_store(markov_store, "csv"))
        z.writestr("llm_triage_input/bipartite_top_candidates.csv", _safe_csv_from_store(markov_store, "candidate_csv"))
        z.writestr("llm_triage_input/projection_diffusion_results.csv", _safe_csv_from_store(projection_diffusion_store, "csv"))
        z.writestr("llm_triage_input/projection_top_candidates.csv", _safe_csv_from_store(projection_diffusion_store, "candidate_csv"))
        z.writestr("llm_triage_input/consensus_candidates.csv", _safe_csv_from_store(consensus_store, "csv"))

    return buf.getvalue()

# ----------------------------
# Callbacks
# ----------------------------

@app.callback(
    Output("store-raw-df", "data"),
    Output("raw-upload-status", "children"),
    Output("item-col", "options"),
    Output("group-col", "options"),
    Output("weight-col", "options"),
    Input("upload-raw", "contents"),
    prevent_initial_call=True,
)
def on_raw_upload(contents):
    try:
        df = parse_upload(contents)
        cols = [{"label": c, "value": c} for c in df.columns]
        status = html.Span(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols", style={"color": "#065f46"})
        return df.to_json(date_format="iso", orient="split"), status, cols, cols, ([{"label": "None", "value": ""}] + cols)
    except Exception as e:
        status = html.Span(f"Upload error: {e}", style={"color": "#b91c1c"})
        return None, status, [], [], []


@app.callback(
    Output("store-raw-df", "data", allow_duplicate=True),
    Output("raw-upload-status", "children", allow_duplicate=True),
    Output("preset-status", "children"),
    Output("item-col", "options", allow_duplicate=True),
    Output("group-col", "options", allow_duplicate=True),
    Output("weight-col", "options", allow_duplicate=True),
    Output("item-col", "value", allow_duplicate=True),
    Output("group-col", "value", allow_duplicate=True),
    Output("weight-col", "value", allow_duplicate=True),
    Input("btn-apply-preset", "n_clicks"),
    State("store-raw-df", "data"),
    State("column-preset", "value"),
    prevent_initial_call=True,
    running=[
        (Output("preset-status", "children"), "Applying column preset...", ""),
        (Output("btn-apply-preset", "disabled"), True, False),
    ],
)
def on_apply_column_preset(n_clicks, raw_json, preset):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if not raw_json:
        msg = html.Span("Upload a CSV or load the demo dataset first.", style={"color": "#b91c1c"})
        return no_update, no_update, msg, no_update, no_update, no_update, no_update, no_update, no_update

    try:
        df = pd.read_json(StringIO(raw_json), orient="split")
        mapped_df, item_col, group_col, weight_col, preset_msg = apply_column_mapping_preset(df, preset)
        cols = [{"label": c, "value": c} for c in mapped_df.columns]

        upload_status = html.Span(
            f"Preset-ready table: {mapped_df.shape[0]:,} rows × {mapped_df.shape[1]} cols",
            style={"color": "#065f46"},
        )
        preset_status = html.Span(preset_msg, style={"color": "#065f46"})

        return (
            mapped_df.to_json(date_format="iso", orient="split"),
            upload_status,
            preset_status,
            cols,
            cols,
            [{"label": "None", "value": ""}] + cols,
            item_col,
            group_col,
            weight_col or "",
        )

    except Exception as e:
        msg = html.Span(f"Preset error: {e}", style={"color": "#b91c1c"})
        return no_update, no_update, msg, no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("store-raw-df", "data", allow_duplicate=True),
    Output("raw-upload-status", "children", allow_duplicate=True),
    Output("demo-status", "children"),
    Output("item-col", "options", allow_duplicate=True),
    Output("group-col", "options", allow_duplicate=True),
    Output("weight-col", "options", allow_duplicate=True),
    Output("item-col", "value"),
    Output("group-col", "value"),
    Output("weight-col", "value"),
    Input("btn-load-demo", "n_clicks"),
    prevent_initial_call=True,
    running=[
        (Output("demo-status", "children"), "Loading demo dataset and expanding gene lists...", ""),
        (Output("btn-load-demo", "disabled"), True, False),
    ],
)
def on_load_demo(n_clicks):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    try:
        df = load_demo_dataframe()
        cols = [{"label": c, "value": c} for c in df.columns]

        status = html.Span(
            f"Demo loaded: {df.shape[0]:,} gene-term edges from "
            f"{df['gene'].nunique():,} genes/items and {df['term'].nunique():,} pathways/terms.",
            style={"color": "#065f46"},
        )

        demo_note = (
            "Next step: click “Preprocess / Build graph” to generate the main bipartite network."
        )

        return (
            df.to_json(date_format="iso", orient="split"),
            status,
            demo_note,
            cols,
            cols,
            [{"label": "None", "value": ""}] + cols,
            "gene",
            "term",
            "adjusted_pvalue",
        )

    except Exception as e:
        err = html.Span(f"Demo load error: {e}", style={"color": "#b91c1c"})
        return None, err, "", [], [], [], None, None, None


@app.callback(
    Output("store-graph", "data"),
    Output("store-nodes-edges-df", "data"),
    Output("plot-hint", "children"),
    Output("graph-size-warning", "children"),
    Input("btn-build", "n_clicks"),
    State("store-raw-df", "data"),
    State("item-col", "value"),
    State("group-col", "value"),
    State("weight-col", "value"),
    prevent_initial_call=True,
    running=[
        (Output("plot-hint", "children"), "Building the main gene ↔ pathway network...", ""),
        (Output("btn-build", "disabled"), True, False),
    ],
)
def build_graph(n_clicks, raw_json, item_col, group_col, weight_col):
    if not raw_json:
        return no_update, no_update, "Upload a raw CSV first.", ""

    if not item_col or not group_col:
        return no_update, no_update, "Select the item and group columns, then click build.", ""

    df = pd.read_json(StringIO(raw_json), orient="split")
    wcol = weight_col if (weight_col and weight_col in df.columns) else None

    g = build_bipartite_graph(df, item_col=item_col, group_col=group_col, weight_col=wcol)

    nodes = [{"id": n, **d} for n, d in g.nodes(data=True)]
    edges = [{"source": u, "target": v, **ed} for u, v, ed in g.edges(data=True)]

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()
    hint = f"Graph built: {node_count:,} nodes, {edge_count:,} edges. Use sidebar controls to explore."

    warning = graph_size_warning_component(node_count, edge_count, context="raw graph")

    return (
        {"nodes": nodes, "edges": edges},
        {"nodes_csv": nodes_df.to_csv(index=False), "edges_csv": edges_df.to_csv(index=False)},
        hint,
        warning,
    )


@app.callback(
    Output("network-graph", "figure"),
    Output("stats-cards", "children"),
    Output("top-groups-table", "data"),
    Output("top-items-table", "data"),
    Input("store-graph", "data"),
    Input("search", "value"),
    Input("min-degree", "value"),
    Input("min-weight", "value"),
    Input("max-groups", "value"),
    Input("layout-mode", "value"),
    Input("store-markov-results", "data"),
    Input("candidate-highlight-toggle", "value"),
    Input("candidate-highlight-count", "value"),
)
def update_plot_and_stats(graph_data, search, min_degree, min_weight, max_groups, layout_mode, markov_results, candidate_highlight_toggle, candidate_highlight_count):
    if not graph_data:
        return go.Figure(), "", [], []

    # rebuild nx graph
    g = nx.Graph()
    for nd in graph_data["nodes"]:
        node_id = nd["id"]
        attrs = nd.copy()
        attrs.pop("id", None)
        g.add_node(node_id, **attrs)
    for ed in graph_data["edges"]:
        u, v = ed["source"], ed["target"]
        attrs = ed.copy()
        attrs.pop("source", None)
        attrs.pop("target", None)
        g.add_edge(u, v, **attrs)

    largest_component_only = "lcc" in []
    show_labels = "labels" in []

    sg = subgraph_filter(
        g,
        search=search or "",
        min_degree=int(min_degree or 0),
        min_weight=float(min_weight or 0),
        max_groups=int(max_groups or 50),
        largest_component_only=largest_component_only,
    )

    if layout_mode == "force":
        pos = nx.spring_layout(sg, seed=7, k=1 / math.sqrt(max(1, sg.number_of_nodes())))
    else:
        pos = layout_bipartite_two_column(sg)

    highlight_nodes = {}
    if "highlight" in (candidate_highlight_toggle or []) and markov_results:
        for row in (markov_results.get("candidate_rows") or [])[: int(candidate_highlight_count or 10)]:
            node_id = row.get("node_id")
            if node_id in sg.nodes:
                highlight_nodes[node_id] = row

    fig = make_plotly_network(sg, pos, show_labels=show_labels, highlight_nodes=highlight_nodes)

    st = graph_stats(sg)
    cards = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "10px"},
        children=[
            html.Div([html.Div("Nodes", style={"color": "#6b7280"}), html.H3(f"{st['nodes']:,}")],
                     style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Edges", style={"color": "#6b7280"}), html.H3(f"{st['edges']:,}")],
                     style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Items", style={"color": "#6b7280"}), html.H3(f"{st['items']:,}")],
                     style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Groups", style={"color": "#6b7280"}), html.H3(f"{st['groups']:,}")],
                     style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Components", style={"color": "#6b7280"}), html.H3(f"{st['components']:,}")],
                     style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Largest component", style={"color": "#6b7280"}), html.H3(f"{st['largest_component']:,}")],
                     style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
        ],
    )

    top_groups_data = [{"group": name, "degree": deg} for name, deg in st["top_groups"]]
    top_items_data = [{"item": name, "degree": deg} for name, deg in st["top_items"]]

    return fig, cards, top_groups_data, top_items_data

@app.callback(
    Output("markov-seed-node", "options"),
    Input("store-graph", "data"),
)
def update_markov_seed_options(graph_data):
    """Populate the seed-node dropdown after graph construction."""
    if not graph_data:
        return []

    g = rebuild_graph_from_store(graph_data)
    options = []

    # Put pathways/groups first because that is usually the more useful seed.
    ordered_nodes = sorted(
        g.nodes(data=True),
        key=lambda x: (0 if x[1].get("node_type") == "group" else 1, str(x[1].get("label", x[0])).lower()),
    )

    for node_id, attrs in ordered_nodes:
        label = attrs.get("label", node_id)
        ntype = attrs.get("node_type", "unknown")
        prefix = "Pathway" if ntype == "group" else "Gene"
        options.append({"label": f"{prefix}: {label}", "value": node_id})

    return options


@app.callback(
    Output("markov-status", "children"),
    Output("markov-summary-cards", "children"),
    Output("markov-candidate-pathways-table", "data"),
    Output("markov-top-groups-table", "data"),
    Output("markov-top-items-table", "data"),
    Output("store-markov-results", "data"),
    Input("btn-run-markov", "n_clicks"),
    State("store-graph", "data"),
    State("markov-seed-node", "value"),
    State("markov-alpha", "value"),
    State("markov-ranking-mode", "value"),
    State("markov-top-n", "value"),
    prevent_initial_call=True,
    running=[
        (Output("markov-status", "children"), "Running bipartite signal diffusion. This may take a moment on larger graphs...", ""),
        (Output("btn-run-markov", "disabled"), True, False),
    ],
)
def run_markov_analysis(n_clicks, graph_data, seed_node, alpha, ranking_mode, top_n):
    """Run global or seeded random-walk analysis and fill the Insights tab."""
    if not graph_data:
        return "Build a graph first, then run signal diffusion analysis.", "", [], [], [], None

    g = rebuild_graph_from_store(graph_data)
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return "The graph is empty. Upload data and build the graph first.", "", [], [], [], None

    alpha = float(alpha or 0.85)
    top_n = int(top_n or 30)

    ranking_mode = (ranking_mode or "balanced").lower().strip()
    scores = run_signal_diffusion(g, seed_node=seed_node, alpha=alpha, ranking_mode=ranking_mode)
    rows_all = diffusion_rows(g, scores, top_n=max(top_n * 4, 100), ranking_mode=ranking_mode)
    group_rows, item_rows = split_diffusion_rows(rows_all, top_n=top_n)
    candidate_pathway_rows = candidate_rows(g, rows_all, top_n=top_n, node_type="group")

    seed_label = None
    if seed_node and seed_node in g.nodes:
        seed_label = g.nodes[seed_node].get("label", seed_node)

    mode_text = f"Seeded from: {seed_label}" if seed_label else "Global ranking"
    mode_labels = {
        "balanced": "Balanced — evidence + connectivity",
        "evidence": "Evidence-weighted",
        "connectivity": "Connectivity-weighted",
    }
    ranking_label = mode_labels.get(ranking_mode, ranking_mode)
    status = f"Signal diffusion complete — {mode_text}. Alpha={alpha:.2f}. Ranking mode: {ranking_label}."

    top_node = rows_all[0] if rows_all else None
    top_group = group_rows[0] if group_rows else None
    top_item = item_rows[0] if item_rows else None

    cards = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "10px"},
        children=[
            html.Div(
                [html.Div("Mode", style={"color": "#6b7280"}), html.H4(mode_text)],
                style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"},
            ),
            html.Div(
                [html.Div("Ranking mode", style={"color": "#6b7280"}), html.H4(ranking_label)],
                style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"},
            ),
            html.Div(
                [html.Div("Top overall node", style={"color": "#6b7280"}), html.H4(top_node["label"] if top_node else "—")],
                style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"},
            ),
            html.Div(
                [html.Div("Top pathway / term", style={"color": "#6b7280"}), html.H4(top_group["label"] if top_group else "—")],
                style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"},
            ),
            html.Div(
                [html.Div("Top gene / item", style={"color": "#6b7280"}), html.H4(top_item["label"] if top_item else "—")],
                style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"},
            ),
        ],
    )

    export_frames = []
    for section_name, section_rows in [
        ("top_followup_candidates_pathways_terms", candidate_pathway_rows),
        ("top_pathways_terms_by_diffusion", group_rows),
        ("top_genes_items_by_diffusion", item_rows),
        ("top_overall_nodes_by_diffusion", rows_all),
    ]:
        section_df = pd.DataFrame(section_rows)
        if section_df.empty:
            continue
        section_df.insert(0, "result_section", section_name)
        export_frames.append(section_df)

    export_df = pd.concat(export_frames, ignore_index=True, sort=False) if export_frames else pd.DataFrame()

    candidates_df = pd.DataFrame(candidate_pathway_rows)
    export_payload = {
        "csv": export_df.to_csv(index=False),
        "candidate_csv": candidates_df.to_csv(index=False),
        "candidate_rows": candidate_pathway_rows,
        "mode": "seeded" if seed_label else "global",
        "seed": seed_label or "",
        "alpha": alpha,
        "ranking_mode": ranking_mode,
    }

    return status, cards, candidate_pathway_rows, group_rows, item_rows, export_payload



@app.callback(
    Output("store-projection-graph", "data"),
    Output("store-projection-export", "data"),
    Output("projection-status", "children"),
    Output("projection-ranking-table", "data"),
    Input("btn-build-projection", "n_clicks"),
    State("store-graph", "data"),
    State("projection-method", "value"),
    State("projection-top-n", "value"),
    prevent_initial_call=True,
    running=[
        (Output("projection-status", "children"), "Building the pathway-overlap projection network...", ""),
        (Output("btn-build-projection", "disabled"), True, False),
    ],
)
def build_projection_network(n_clicks, graph_data, method, top_n):
    """Build a separate pathway-only projection graph from the current membership graph."""
    if not graph_data:
        return no_update, no_update, "Build the main gene ↔ pathway graph first.", []

    g = rebuild_graph_from_store(graph_data)
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return no_update, no_update, "The main graph is empty. Upload data and build the graph first.", []

    pg = build_pathway_projection_graph(g, method=method or "jaccard")
    if pg.number_of_nodes() == 0 or pg.number_of_edges() == 0:
        empty_store = projection_graph_to_store(pg)
        return empty_store, {"nodes_csv": "", "edges_csv": ""}, "No pathway overlaps found. The current pathways do not share genes/items.", []

    ranking_rows = projection_stats_rows(pg, top_n=int(top_n or 30))
    store_graph = projection_graph_to_store(pg)

    nodes_df = pd.DataFrame(store_graph["nodes"])
    edges_df = pd.DataFrame(store_graph["edges"])
    export_store = {
        "nodes_csv": nodes_df.to_csv(index=False),
        "edges_csv": edges_df.to_csv(index=False),
        "method": method or "jaccard",
    }

    method_label = {
        "jaccard": "Jaccard similarity",
        "shared_count": "shared gene count",
        "weighted_shared": "weighted shared support",
    }.get(method or "jaccard", method or "jaccard")

    status = (
        f"Projection built with {method_label}: "
        f"{pg.number_of_nodes():,} pathway nodes and {pg.number_of_edges():,} pathway-overlap edges."
    )
    return store_graph, export_store, status, ranking_rows


@app.callback(
    Output("projection-graph", "figure"),
    Input("store-projection-graph", "data"),
    Input("projection-show-labels", "value"),
)
def update_projection_graph(projection_data, show_labels_values):
    """Render the pathway-only projection graph."""
    if not projection_data:
        return make_projection_figure(nx.Graph(), show_labels=True)
    pg = rebuild_projection_from_store(projection_data)
    show_labels = "labels" in (show_labels_values or [])
    return make_projection_figure(pg, show_labels=show_labels)


@app.callback(
    Output("projection-diffusion-seed-node", "options"),
    Input("store-projection-graph", "data"),
)
def update_projection_diffusion_seed_options(projection_data):
    if not projection_data:
        return []
    pg = rebuild_projection_from_store(projection_data)
    ordered_nodes = sorted(pg.nodes(data=True), key=lambda x: str(x[1].get("label", x[0])).lower())
    return [{"label": f"Pathway: {attrs.get('label', node_id)}", "value": node_id} for node_id, attrs in ordered_nodes]


@app.callback(
    Output("projection-diffusion-status", "children"),
    Output("projection-diffusion-summary-cards", "children"),
    Output("projection-diffusion-candidates-table", "data"),
    Output("projection-diffusion-rankings-table", "data"),
    Output("store-projection-diffusion-results", "data"),
    Input("btn-run-projection-diffusion", "n_clicks"),
    State("store-projection-graph", "data"),
    State("projection-diffusion-seed-node", "value"),
    State("projection-diffusion-alpha", "value"),
    State("projection-diffusion-ranking-mode", "value"),
    State("projection-diffusion-top-n", "value"),
    prevent_initial_call=True,
    running=[
        (Output("projection-diffusion-status", "children"), "Running projection diffusion across the pathway-overlap graph...", ""),
        (Output("btn-run-projection-diffusion", "disabled"), True, False),
    ],
)
def run_projection_diffusion_analysis(n_clicks, projection_data, seed_node, alpha, ranking_mode, top_n):
    if not projection_data:
        return "Build the projection network first, then run projection diffusion analysis.", "", [], [], None
    pg = rebuild_projection_from_store(projection_data)
    if pg.number_of_nodes() == 0 or pg.number_of_edges() == 0:
        return "The projection graph is empty. Build a projection network with pathway overlaps first.", "", [], [], None
    alpha = float(alpha or 0.85)
    top_n = int(top_n or 30)
    ranking_mode = (ranking_mode or "balanced").lower().strip()
    scores = run_projection_diffusion(pg, seed_node=seed_node, alpha=alpha, ranking_mode=ranking_mode)
    rows_all = projection_diffusion_rows(pg, scores, top_n=max(top_n * 4, 100), ranking_mode=ranking_mode)
    candidate_rows_out = projection_candidate_rows(pg, rows_all, top_n=top_n)
    ranking_rows = rows_all[:top_n]
    seed_label = pg.nodes[seed_node].get("label", seed_node) if seed_node and seed_node in pg.nodes else None
    mode_text = f"Seeded from: {seed_label}" if seed_label else "Global projection ranking"
    mode_labels = {"balanced": "Balanced — overlap + connectivity", "evidence": "Overlap-weighted", "connectivity": "Connectivity-weighted"}
    ranking_label = mode_labels.get(ranking_mode, ranking_mode)
    status = f"Projection diffusion complete — {mode_text}. Alpha={alpha:.2f}. Ranking mode: {ranking_label}."
    top_node = ranking_rows[0] if ranking_rows else None
    top_candidate = candidate_rows_out[0] if candidate_rows_out else None
    cards = html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "10px"}, children=[
        html.Div([html.Div("Mode", style={"color": "#6b7280"}), html.H4(mode_text)], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
        html.Div([html.Div("Ranking mode", style={"color": "#6b7280"}), html.H4(ranking_label)], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
        html.Div([html.Div("Top projected pathway", style={"color": "#6b7280"}), html.H4(top_node["label"] if top_node else "—")], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
        html.Div([html.Div("Top follow-up candidate", style={"color": "#6b7280"}), html.H4(top_candidate["label"] if top_candidate else "—")], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
    ])
    export_frames = []
    for section_name, section_rows in [("top_followup_candidates_projection_pathways", candidate_rows_out), ("top_pathways_by_projection_diffusion", ranking_rows), ("top_overall_projection_diffusion", rows_all)]:
        section_df = pd.DataFrame(section_rows)
        if section_df.empty:
            continue
        section_df.insert(0, "result_section", section_name)
        export_frames.append(section_df)
    export_df = pd.concat(export_frames, ignore_index=True, sort=False) if export_frames else pd.DataFrame()
    candidates_df = pd.DataFrame(candidate_rows_out)
    export_payload = {"csv": export_df.to_csv(index=False), "candidate_csv": candidates_df.to_csv(index=False), "candidate_rows": candidate_rows_out, "mode": "seeded" if seed_label else "global", "seed": seed_label or "", "alpha": alpha, "ranking_mode": ranking_mode}
    return status, cards, candidate_rows_out, ranking_rows, export_payload


@app.callback(
    Output("dl-projection-nodes", "data"),
    Input("btn-dl-projection-nodes", "n_clicks"),
    State("store-projection-export", "data"),
    prevent_initial_call=True,
)
def download_projection_nodes(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store.get("nodes_csv", ""), filename="projected_pathway_nodes.csv")


@app.callback(
    Output("dl-projection-edges", "data"),
    Input("btn-dl-projection-edges", "n_clicks"),
    State("store-projection-export", "data"),
    prevent_initial_call=True,
)
def download_projection_edges(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store.get("edges_csv", ""), filename="projected_pathway_edges.csv")


@app.callback(
    Output("dl-projection-diffusion", "data"),
    Input("btn-dl-projection-diffusion", "n_clicks"),
    State("store-projection-diffusion-results", "data"),
    prevent_initial_call=True,
)
def download_projection_diffusion_results(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store.get("csv", ""), filename="projection_diffusion_results.csv")


@app.callback(
    Output("dl-projection-candidates", "data"),
    Input("btn-dl-projection-candidates", "n_clicks"),
    State("store-projection-diffusion-results", "data"),
    prevent_initial_call=True,
)
def download_projection_candidates(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store.get("candidate_csv", ""), filename="projection_top_candidates.csv")


@app.callback(
    Output("dl-markov", "data"),
    Input("btn-dl-markov", "n_clicks"),
    State("store-markov-results", "data"),
    prevent_initial_call=True,
)
def download_markov_results(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store["csv"], filename="diffusion_results.csv")


@app.callback(
    Output("dl-candidates", "data"),
    Input("btn-dl-candidates", "n_clicks"),
    State("store-markov-results", "data"),
    prevent_initial_call=True,
)
def download_candidate_results(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store.get("candidate_csv", ""), filename="top_candidates.csv")


@app.callback(
    Output("dl-nodes", "data"),
    Input("btn-dl-nodes", "n_clicks"),
    State("store-nodes-edges-df", "data"),
    prevent_initial_call=True,
)
def download_nodes(n, store):
    if not store:
        return no_update
    return dict(content=store["nodes_csv"], filename="nodes.csv")


@app.callback(
    Output("dl-edges", "data"),
    Input("btn-dl-edges", "n_clicks"),
    State("store-nodes-edges-df", "data"),
    prevent_initial_call=True,
)
def download_edges(n, store):
    if not store:
        return no_update
    return dict(content=store["edges_csv"], filename="edges.csv")


@app.callback(
    Output("consensus-status", "children"),
    Output("consensus-summary-cards", "children"),
    Output("consensus-candidates-table", "data"),
    Output("store-consensus-results", "data"),
    Input("btn-build-consensus", "n_clicks"),
    State("store-markov-results", "data"),
    State("store-projection-diffusion-results", "data"),
    State("consensus-top-n", "value"),
    prevent_initial_call=True,
    running=[
        (Output("consensus-status", "children"), "Building consensus candidates from bipartite and projection rankings...", ""),
        (Output("btn-build-consensus", "disabled"), True, False),
    ],
)
def build_consensus_candidates(n_clicks, bipartite_store, projection_store, top_n):
    if not bipartite_store:
        return "Run Bipartite Diffusion first, then build consensus candidates.", "", [], None
    if not projection_store:
        return "Run Projection Diffusion first, then build consensus candidates.", "", [], None

    top_n = int(top_n or 30)
    rows = consensus_candidate_rows(bipartite_store, projection_store, top_n=top_n)
    if not rows:
        return "No consensus candidates could be built. Make sure both diffusion candidate tables contain pathway results.", "", [], None

    top = rows[0]
    present_in_both = sum(1 for r in rows if r.get("bipartite_followup_score", 0) and r.get("projection_followup_score", 0))
    status = f"Consensus complete: {len(rows):,} candidates ranked. {present_in_both:,} appear in both bipartite and projection candidate lists."
    cards = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "10px"},
        children=[
            html.Div([html.Div("Top consensus candidate", style={"color": "#6b7280"}), html.H4(top.get("label", "—"))], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Consensus score", style={"color": "#6b7280"}), html.H4(top.get("consensus_score", "—"))], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Bipartite rank", style={"color": "#6b7280"}), html.H4(top.get("bipartite_candidate_rank", "—"))], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
            html.Div([html.Div("Projection rank", style={"color": "#6b7280"}), html.H4(top.get("projection_candidate_rank", "—"))], style={"padding": "12px", "borderRadius": "12px", "background": "#f8fafc"}),
        ],
    )
    export_payload = {"csv": pd.DataFrame(rows).to_csv(index=False), "candidate_rows": rows}
    return status, cards, rows, export_payload


@app.callback(
    Output("dl-consensus", "data"),
    Input("btn-dl-consensus", "n_clicks"),
    State("store-consensus-results", "data"),
    prevent_initial_call=True,
)
def download_consensus_candidates(n_clicks, store):
    if not store:
        return no_update
    return dict(content=store.get("csv", ""), filename="consensus_candidates.csv")


@app.callback(
    Output("dl-report-bundle", "data"),
    Output("report-bundle-status", "children"),
    Input("btn-dl-report-bundle", "n_clicks"),
    State("store-raw-df", "data"),
    State("store-graph", "data"),
    State("store-nodes-edges-df", "data"),
    State("store-projection-graph", "data"),
    State("store-projection-export", "data"),
    State("store-markov-results", "data"),
    State("store-projection-diffusion-results", "data"),
    State("store-consensus-results", "data"),
    State("item-col", "value"),
    State("group-col", "value"),
    State("weight-col", "value"),
    State("search", "value"),
    State("min-degree", "value"),
    State("min-weight", "value"),
    State("max-groups", "value"),
    State("layout-mode", "value"),
    State("edge-style", "value"),
    State("edge-width-range", "value"),
    State("edge-weight-range", "value"),
    State("markov-alpha", "value"),
    State("markov-ranking-mode", "value"),
    State("markov-top-n", "value"),
    State("projection-method", "value"),
    State("projection-top-n", "value"),
    State("projection-show-labels", "value"),
    State("projection-diffusion-alpha", "value"),
    State("projection-diffusion-ranking-mode", "value"),
    State("projection-diffusion-top-n", "value"),
    State("consensus-top-n", "value"),
    State("candidate-highlight-toggle", "value"),
    State("candidate-highlight-count", "value"),
    prevent_initial_call=True,
    running=[
        (Output("report-bundle-running-status", "children"), "Generating report bundle. Please wait — graph images and CSV exports can take a moment...", ""),
        (Output("btn-dl-report-bundle", "disabled"), True, False),
    ],
)
def download_report_bundle(
    n_clicks,
    raw_json,
    graph_data,
    nodes_edges_store,
    projection_data,
    projection_export_store,
    markov_store,
    projection_diffusion_store,
    consensus_store,
    item_col,
    group_col,
    weight_col,
    search,
    min_degree,
    min_weight,
    max_groups,
    layout_mode,
    edge_style,
    edge_width_range,
    edge_weight_range,
    markov_alpha,
    markov_ranking_mode,
    markov_top_n,
    projection_method,
    projection_top_n,
    projection_show_labels,
    projection_diffusion_alpha,
    projection_diffusion_ranking_mode,
    projection_diffusion_top_n,
    consensus_top_n,
    candidate_highlight_toggle,
    candidate_highlight_count,
):
    if not n_clicks:
        return no_update, ""

    if not graph_data:
        return no_update, "Build the main graph before downloading the report bundle."

    mapped_columns = {
        "item_col": item_col,
        "group_col": group_col,
        "weight_col": weight_col,
    }

    settings = {
        "main_graph": {
            "search": search or "",
            "min_degree": int(min_degree or 0),
            "min_weight": float(min_weight or 0),
            "max_groups": int(max_groups or 50),
            "layout_mode": layout_mode or "bipartite",
            "edge_style": edge_style or [],
            "edge_width_range": edge_width_range,
            "edge_weight_range": edge_weight_range,
        },
        "bipartite_diffusion": {
            "alpha": markov_alpha,
            "ranking_mode": markov_ranking_mode,
            "top_n": markov_top_n,
        },
        "projection": {
            "method": projection_method,
            "top_n": projection_top_n,
            "show_labels": projection_show_labels,
        },
        "projection_diffusion": {
            "alpha": projection_diffusion_alpha,
            "ranking_mode": projection_diffusion_ranking_mode,
            "top_n": projection_diffusion_top_n,
        },
        "consensus": {
            "top_n": consensus_top_n,
        },
        "candidate_highlighting": {
            "enabled": candidate_highlight_toggle,
            "count": candidate_highlight_count,
        },
        "important_note": "Diffusion, follow-up, and consensus scores are prioritization scores, not statistical p-values.",
    }

    zip_bytes = build_report_bundle_zip(
        raw_json=raw_json,
        graph_data=graph_data,
        nodes_edges_store=nodes_edges_store,
        projection_data=projection_data,
        projection_export_store=projection_export_store,
        markov_store=markov_store,
        projection_diffusion_store=projection_diffusion_store,
        consensus_store=consensus_store,
        mapped_columns=mapped_columns,
        settings=settings,
        search=search,
        min_degree=min_degree,
        min_weight=min_weight,
        max_groups=max_groups,
        layout_mode=layout_mode,
        candidate_highlight_toggle=candidate_highlight_toggle,
        candidate_highlight_count=candidate_highlight_count,
        projection_show_labels=projection_show_labels,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enrichment_network_studio_report_bundle_{timestamp}.zip"

    # Explicit dcc.Download payload for a zip file.
    payload = {
        "content": base64.b64encode(zip_bytes).decode("utf-8"),
        "filename": filename,
        "type": "application/zip",
        "base64": True,
    }
    return payload, f"Report bundle ready: {filename}"



@app.callback(
    Output("dl-llm-triage-bundle", "data"),
    Output("llm-triage-bundle-status", "children"),
    Input("btn-dl-llm-triage-bundle", "n_clicks"),
    State("store-raw-df", "data"),
    State("store-graph", "data"),
    State("store-nodes-edges-df", "data"),
    State("store-projection-graph", "data"),
    State("store-projection-export", "data"),
    State("store-markov-results", "data"),
    State("store-projection-diffusion-results", "data"),
    State("store-consensus-results", "data"),
    State("item-col", "value"),
    State("group-col", "value"),
    State("weight-col", "value"),
    State("search", "value"),
    State("min-degree", "value"),
    State("min-weight", "value"),
    State("max-groups", "value"),
    State("layout-mode", "value"),
    State("edge-style", "value"),
    State("edge-width-range", "value"),
    State("edge-weight-range", "value"),
    State("markov-alpha", "value"),
    State("markov-ranking-mode", "value"),
    State("markov-top-n", "value"),
    State("projection-method", "value"),
    State("projection-top-n", "value"),
    State("projection-show-labels", "value"),
    State("projection-diffusion-alpha", "value"),
    State("projection-diffusion-ranking-mode", "value"),
    State("projection-diffusion-top-n", "value"),
    State("consensus-top-n", "value"),
    prevent_initial_call=True,
    running=[
        (Output("llm-triage-bundle-running-status", "children"), "Generating LLM Triage input bundle. Please wait...", ""),
        (Output("btn-dl-llm-triage-bundle", "disabled"), True, False),
    ],
)
def download_llm_triage_bundle(
    n_clicks,
    raw_json,
    graph_data,
    nodes_edges_store,
    projection_data,
    projection_export_store,
    markov_store,
    projection_diffusion_store,
    consensus_store,
    item_col,
    group_col,
    weight_col,
    search,
    min_degree,
    min_weight,
    max_groups,
    layout_mode,
    edge_style,
    edge_width_range,
    edge_weight_range,
    markov_alpha,
    markov_ranking_mode,
    markov_top_n,
    projection_method,
    projection_top_n,
    projection_show_labels,
    projection_diffusion_alpha,
    projection_diffusion_ranking_mode,
    projection_diffusion_top_n,
    consensus_top_n,
):
    if not n_clicks:
        return no_update, ""

    if not graph_data:
        return no_update, "Build the main graph before exporting the LLM Triage bundle."

    mapped_columns = {
        "item_col": item_col,
        "group_col": group_col,
        "weight_col": weight_col,
    }

    settings = {
        "main_graph": {
            "search": search or "",
            "min_degree": int(min_degree or 0),
            "min_weight": float(min_weight or 0),
            "max_groups": int(max_groups or 50),
            "layout_mode": layout_mode or "bipartite",
            "edge_style": edge_style or [],
            "edge_width_range": edge_width_range,
            "edge_weight_range": edge_weight_range,
        },
        "bipartite_diffusion": {
            "alpha": markov_alpha,
            "ranking_mode": markov_ranking_mode,
            "top_n": markov_top_n,
        },
        "projection": {
            "method": projection_method,
            "top_n": projection_top_n,
            "show_labels": projection_show_labels,
        },
        "projection_diffusion": {
            "alpha": projection_diffusion_alpha,
            "ranking_mode": projection_diffusion_ranking_mode,
            "top_n": projection_diffusion_top_n,
        },
        "consensus": {
            "top_n": consensus_top_n,
        },
        "important_note": "This export does not run an LLM or spend API credits. Scores are prioritization scores, not p-values.",
    }

    zip_bytes = build_llm_triage_bundle_zip(
        raw_json=raw_json,
        graph_data=graph_data,
        nodes_edges_store=nodes_edges_store,
        projection_data=projection_data,
        projection_export_store=projection_export_store,
        markov_store=markov_store,
        projection_diffusion_store=projection_diffusion_store,
        consensus_store=consensus_store,
        mapped_columns=mapped_columns,
        settings=settings,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_triage_input_bundle_{timestamp}.zip"

    payload = {
        "content": base64.b64encode(zip_bytes).decode("utf-8"),
        "filename": filename,
        "type": "application/zip",
        "base64": True,
    }
    return payload, f"LLM Triage input bundle ready: {filename}"


@app.callback(
    Output("dl-main-svg", "data"),
    Input("btn-dl-main-svg", "n_clicks"),
    State("store-graph", "data"),
    State("search", "value"),
    State("min-degree", "value"),
    State("min-weight", "value"),
    State("max-groups", "value"),
    State("layout-mode", "value"),
    State("store-markov-results", "data"),
    State("candidate-highlight-toggle", "value"),
    State("candidate-highlight-count", "value"),
    prevent_initial_call=True,
)
def download_main_graph_svg(n_clicks, graph_data, search, min_degree, min_weight, max_groups, layout_mode, markov_results, candidate_highlight_toggle, candidate_highlight_count):
    if not graph_data:
        return no_update
    fig = build_main_graph_figure_for_export(graph_data, search, min_degree, min_weight, max_groups, layout_mode, markov_results, candidate_highlight_toggle, candidate_highlight_count)
    return _download_plotly_figure(fig, "svg", "main_bipartite_network.svg")


@app.callback(
    Output("dl-main-pdf", "data"),
    Input("btn-dl-main-pdf", "n_clicks"),
    State("store-graph", "data"),
    State("search", "value"),
    State("min-degree", "value"),
    State("min-weight", "value"),
    State("max-groups", "value"),
    State("layout-mode", "value"),
    State("store-markov-results", "data"),
    State("candidate-highlight-toggle", "value"),
    State("candidate-highlight-count", "value"),
    prevent_initial_call=True,
)
def download_main_graph_pdf(n_clicks, graph_data, search, min_degree, min_weight, max_groups, layout_mode, markov_results, candidate_highlight_toggle, candidate_highlight_count):
    if not graph_data:
        return no_update
    fig = build_main_graph_figure_for_export(graph_data, search, min_degree, min_weight, max_groups, layout_mode, markov_results, candidate_highlight_toggle, candidate_highlight_count)
    return _download_plotly_figure(fig, "pdf", "main_bipartite_network.pdf")


@app.callback(
    Output("dl-projection-svg", "data"),
    Input("btn-dl-projection-svg", "n_clicks"),
    State("store-projection-graph", "data"),
    State("projection-show-labels", "value"),
    prevent_initial_call=True,
)
def download_projection_graph_svg(n_clicks, projection_data, show_labels_values):
    if not projection_data:
        return no_update
    pg = rebuild_projection_from_store(projection_data)
    show_labels = "labels" in (show_labels_values or [])
    fig = make_projection_figure(pg, show_labels=show_labels)
    fig.update_layout(width=1400, height=1000)
    return _download_plotly_figure(fig, "svg", "pathway_projection_network.svg")


@app.callback(
    Output("dl-projection-pdf", "data"),
    Input("btn-dl-projection-pdf", "n_clicks"),
    State("store-projection-graph", "data"),
    State("projection-show-labels", "value"),
    prevent_initial_call=True,
)
def download_projection_graph_pdf(n_clicks, projection_data, show_labels_values):
    if not projection_data:
        return no_update
    pg = rebuild_projection_from_store(projection_data)
    show_labels = "labels" in (show_labels_values or [])
    fig = make_projection_figure(pg, show_labels=show_labels)
    fig.update_layout(width=1400, height=1000)
    return _download_plotly_figure(fig, "pdf", "pathway_projection_network.pdf")


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)





