import networkx as nx

from signed_graphrag.balance import analyze_balance, signed_triangles, two_camp_partition


def make(edges):
    g = nx.Graph()
    for u, v, s in edges:
        g.add_edge(u, v, sign=s, weight=float(s))
    return g


def test_balanced_patterns():
    for signs in [(1, 1, 1), (1, -1, -1)]:
        g = make([("A", "B", signs[0]), ("B", "C", signs[1]), ("A", "C", signs[2])])
        (t,) = signed_triangles(g)
        assert t.balanced


def test_unbalanced_patterns():
    for signs in [(1, 1, -1), (-1, -1, -1)]:
        g = make([("A", "B", signs[0]), ("B", "C", signs[1]), ("A", "C", signs[2])])
        (t,) = signed_triangles(g)
        assert not t.balanced


def test_two_perfect_camps_have_zero_frustration():
    camp1, camp2 = ["a", "b", "c"], ["x", "y", "z"]
    edges = [(u, v, 1) for grp in (camp1, camp2) for i, u in enumerate(grp) for v in grp[i + 1:]]
    edges += [(u, v, -1) for u in camp1 for v in camp2]
    g = make(edges)
    rep = analyze_balance(g)
    assert rep.balance_ratio == 1.0
    assert rep.frustration == 0
    camps = rep.camps
    assert len({camps[n] for n in camp1}) == 1
    assert len({camps[n] for n in camp2}) == 1
    assert camps["a"] != camps["x"]


def test_unbalanced_edge_is_flagged():
    g = make([("A", "B", 1), ("B", "C", 1), ("A", "C", -1)])
    rep = analyze_balance(g)
    assert rep.balance_ratio == 0.0
    assert rep.frustration == 1
    assert rep.suspicious_edges


def test_empty_graph():
    rep = analyze_balance(nx.Graph())
    assert rep.balance_ratio == 1.0 and rep.frustration == 0
    assert two_camp_partition(nx.Graph()) == {}
