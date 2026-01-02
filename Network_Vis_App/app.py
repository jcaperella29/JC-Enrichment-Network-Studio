import io
import math
from typing import Dict, Tuple
from typing import Dict
import networkx as nx
import numpy as np
import base64
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

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
) -> go.Figure:
    """
    Draw a bipartite-ish network with:
      - edges colored by term (group node)
      - optional edge thickness scaling by weight
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
    node_x, node_y, node_text, node_hover, node_size, node_color = [], [], [], [], [], []

    for n, d in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        label = str(d.get("label", n))
        ntype = d.get("node_type", "unknown")
        deg = g.degree(n)

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
            line=dict(width=1, color="rgba(0,0,0,0.25)"),
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
# Dash App
# ----------------------------
app = Dash(__name__)
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

                html.Div(
                    style={"marginTop": "12px"},
                    children=[
                        html.Label("Item column (genes)"),
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
    Output("store-graph", "data"),
    Output("store-nodes-edges-df", "data"),
    Output("plot-hint", "children"),
    Input("btn-build", "n_clicks"),
    State("store-raw-df", "data"),
    State("item-col", "value"),
    State("group-col", "value"),
    State("weight-col", "value"),
    prevent_initial_call=True,
)
def build_graph(n_clicks, raw_json, item_col, group_col, weight_col):
    if not raw_json:
        return no_update, no_update, "Upload a raw CSV first."

    if not item_col or not group_col:
        return no_update, no_update, "Select the item and group columns, then click build."

    df = pd.read_json(raw_json, orient="split")
    wcol = weight_col if (weight_col and weight_col in df.columns) else None

    g = build_bipartite_graph(df, item_col=item_col, group_col=group_col, weight_col=wcol)

    nodes = [{"id": n, **d} for n, d in g.nodes(data=True)]
    edges = [{"source": u, "target": v, **ed} for u, v, ed in g.edges(data=True)]

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)

    hint = f"Graph built: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges. Use sidebar controls to explore."

    return (
        {"nodes": nodes, "edges": edges},
        {"nodes_csv": nodes_df.to_csv(index=False), "edges_csv": edges_df.to_csv(index=False)},
        hint,
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
)
def update_plot_and_stats(graph_data, search, min_degree, min_weight, max_groups, layout_mode):
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

    fig = make_plotly_network(sg, pos, show_labels=show_labels)

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


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)





