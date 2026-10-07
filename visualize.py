"""Draw the signed graph: green = positive, red dashed = negative,
node colour = community (faction)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756",
           "#72B7B2", "#EECA3B", "#9D755D", "#BAB0AC", "#FF9DA6"]


def draw_signed_graph(g: nx.Graph, communities: dict[str, int] | None, path: str | Path,
                      title: str = "Signed knowledge graph") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # layout on positive edges only so allies cluster together
    pos_g = nx.Graph()
    pos_g.add_nodes_from(g.nodes())
    pos_g.add_edges_from((u, v) for u, v, d in g.edges(data=True) if d["sign"] > 0)
    layout = nx.spring_layout(pos_g, seed=7, k=1.1)

    fig, ax = plt.subplots(figsize=(13, 9))
    pos_edges = [(u, v) for u, v, d in g.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in g.edges(data=True) if d["sign"] < 0]
    contested = [(u, v) for u, v, d in g.edges(data=True) if d.get("contested")]

    nx.draw_networkx_edges(g, layout, edgelist=pos_edges, edge_color="#2E8B57", width=2.2, ax=ax)
    nx.draw_networkx_edges(g, layout, edgelist=neg_edges, edge_color="#C0392B", width=1.8,
                           style="dashed", ax=ax)
    if contested:
        nx.draw_networkx_edges(g, layout, edgelist=contested, edge_color="#E6A817", width=4,
                               alpha=0.45, ax=ax)
    colors = [PALETTE[(communities or {}).get(n, 0) % len(PALETTE)] for n in g.nodes()]
    nx.draw_networkx_nodes(g, layout, node_color=colors, node_size=900, edgecolors="white", ax=ax)
    nx.draw_networkx_labels(g, layout, font_size=9, ax=ax)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2E8B57", lw=2.2, label="positive (+)"),
        Line2D([0], [0], color="#C0392B", lw=1.8, ls="--", label="negative (-)"),
        Line2D([0], [0], color="#E6A817", lw=4, alpha=0.45, label="contested"),
    ], loc="lower left", frameon=False)
    ax.set_title(title + " (node colour = faction)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
