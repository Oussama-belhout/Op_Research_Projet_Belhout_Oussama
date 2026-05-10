"""
tests/test_all.py
Tests pour le projet de flot.

Pas de framework — j'utilise juste des asserts et un petit runner manuel à la
fin. C'est plus que suffisant vu la taille du projet et ça permet de lancer
avec un simple `python tests/test_all.py` depuis la racine.

Ce qu'on couvre:
  - flot max sur petit graphe + conservation
  - MCF Bellman-Ford avec un arc à coût négatif
  - MCF Dijkstra/Johnson — doit donner le même résultat que la version BF
  - détection de cycle négatif (vrai positif et vrai négatif)
  - MCF qui avorte proprement face à un cycle négatif
  - stress test sur ~10 noeuds
"""

import os
import sys

# bidouille pour pouvoir lancer les tests depuis n'importe où
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph import ResidualGraph
from src.ford_fulkerson import ford_fulkerson
from src.min_cost_flow import min_cost_flow_bellman_ford, min_cost_flow_dijkstra
from src.negative_cycle import (
    detect_negative_cycle_bellman_ford,
    detect_negative_cycle_dfs,
)


# ---------- helpers ----------

def _conservation_ok(flows, sources, sinks):
    """Vérifie la conservation du flot sur tous les noeuds sauf sources et sinks."""
    nodes = set()
    for (u, v) in flows:
        nodes.add(u); nodes.add(v)
    for v in nodes:
        if v in sources or v in sinks:
            continue
        inflow = sum(f for (a, b), f in flows.items() if b == v)
        outflow = sum(f for (a, b), f in flows.items() if a == v)
        if inflow != outflow:
            return False, v, inflow, outflow
    return True, None, None, None


# ---------- tests ----------

def test_simple_max_flow():
    """4 noeuds. Coupes possibles: {0}={3+2}=5, {0,1}={2+1+2}=5, {0,1,2}={2+3}=5.
    Donc max flow = 5."""
    g = ResidualGraph()
    g.add_edge(0, 1, 3)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 2)
    g.add_edge(2, 3, 3)

    mf, flows, S, T, cut = ford_fulkerson(g, 0, 3)

    assert mf == 5, f"Attendu max flow = 5, obtenu {mf}"
    assert 0 in S and 3 in T, f"S={S}, T={T} — source/sink doivent être séparés"

    # théorème max-flow / min-cut: capacité de la coupe = valeur du flot
    cut_capacity = sum(g.graph[u][v]['capacity'] for (u, v) in cut)
    assert cut_capacity == mf, \
        f"capacité de la coupe = {cut_capacity}, devrait = max flow = {mf}"

    print(f"test_simple_max_flow: OK (max_flow={mf}, cut={cut})")


def test_max_flow_conservation():
    """Vérifie la conservation à chaque noeud intermédiaire."""
    g = ResidualGraph()
    g.add_edge(0, 1, 10)
    g.add_edge(0, 2, 5)
    g.add_edge(1, 3, 7)
    g.add_edge(2, 3, 8)
    g.add_edge(1, 2, 4)

    _, flows, _, _, _ = ford_fulkerson(g, 0, 3)

    ok, v, inn, outn = _conservation_ok(flows, {0}, {3})
    assert ok, f"conservation cassée au noeud {v}: in={inn}, out={outn}"
    print("test_max_flow_conservation: OK")


def test_ford_fulkerson_known_value():
    """Petit graphe parallèle: max flow doit être min(out_s, in_t) = 12."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5)
    g.add_edge(0, 2, 7)
    g.add_edge(1, 3, 6)
    g.add_edge(2, 3, 8)
    mf, _, _, _, _ = ford_fulkerson(g, 0, 3)
    assert mf == 12, f"Attendu 12, obtenu {mf}"
    print("test_ford_fulkerson_known_value: OK")


def test_mcf_bellman_ford_negative_cost():
    """MCF avec un arc à coût négatif — BF doit pouvoir le gérer.
    On demande à pousser 5 unités du source au puits."""
    g = ResidualGraph()
    g.add_edge(0, 1, 4, cost=2)
    g.add_edge(0, 2, 3, cost=1)
    g.add_edge(1, 2, 2, cost=-1)
    g.add_edge(1, 3, 3, cost=4)
    g.add_edge(2, 3, 4, cost=2)

    cost, flow, flows = min_cost_flow_bellman_ford(g, 0, 3, max_flow=5)

    assert cost is not None and flow == 5
    ok, v, inn, outn = _conservation_ok(flows, {0}, {3})
    assert ok, f"conservation cassée au noeud {v}: in={inn}, out={outn}"
    print(f"test_mcf_bellman_ford_negative_cost: OK (cost={cost}, flow={flow})")


def test_mcf_dijkstra_johnson_matches_bf():
    """Les deux algos doivent donner exactement le même coût et le même volume
    de flot sur le même graphe (à coûts positifs). C'est l'unique vraie sanity
    check qu'on a — si ça ne match pas, y'a un bug."""
    def make_graph():
        g = ResidualGraph()
        g.add_edge(0, 1, 5, cost=3)
        g.add_edge(0, 2, 4, cost=2)
        g.add_edge(1, 2, 2, cost=1)
        g.add_edge(1, 3, 3, cost=2)
        g.add_edge(2, 3, 5, cost=4)
        return g

    g1 = make_graph()
    g2 = make_graph()

    c1, f1, _ = min_cost_flow_bellman_ford(g1, 0, 3)
    c2, f2, _ = min_cost_flow_dijkstra(g2, 0, 3)

    assert c1 == c2, f"BF: {c1}, Dijkstra: {c2}"
    assert f1 == f2, f"BF flow: {f1}, Dijkstra flow: {f2}"
    print(f"test_mcf_dijkstra_johnson_matches_bf: OK (cost={c1}, flow={f1})")


def test_mcf_dijkstra_on_negative_arc():
    """Même test mais avec un arc à coût négatif. Johnson doit toujours marcher
    parce que BF initial calcule des potentiels qui rendent tout positif après
    reweighting."""
    def make_graph():
        g = ResidualGraph()
        g.add_edge(0, 1, 4, cost=2)
        g.add_edge(0, 2, 3, cost=1)
        g.add_edge(1, 2, 2, cost=-1)
        g.add_edge(1, 3, 3, cost=4)
        g.add_edge(2, 3, 4, cost=2)
        return g

    g1 = make_graph()
    g2 = make_graph()

    c1, f1, _ = min_cost_flow_bellman_ford(g1, 0, 3, max_flow=5)
    c2, f2, _ = min_cost_flow_dijkstra(g2, 0, 3, max_flow=5)

    assert c1 == c2 and f1 == f2, \
        f"Dijkstra et BF doivent donner le même résultat: BF=({c1},{f1}), DJ=({c2},{f2})"
    print(f"test_mcf_dijkstra_on_negative_arc: OK (cost={c1}, flow={f1})")


def test_negative_cycle_detection_positive():
    """Cycle 2 -> 3 -> 4 -> 2 avec coûts -3, 2, -2 → somme -3. Doit être détecté."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5, cost=1)
    g.add_edge(1, 2, 5, cost=1)
    g.add_edge(2, 3, 5, cost=-3)
    g.add_edge(3, 4, 5, cost=2)
    g.add_edge(4, 2, 5, cost=-2)
    g.add_edge(4, 5, 5, cost=1)

    has_cycle, cycle = detect_negative_cycle_bellman_ford(g)
    assert has_cycle, "le cycle aurait dû être détecté"

    # vérifier que le cycle extrait est bien de poids strictement négatif
    total = 0
    for i in range(len(cycle) - 1):
        total += g.graph[cycle[i]][cycle[i + 1]]['cost']
    assert total < 0, f"poids du cycle extrait = {total}, attendu < 0"
    print(f"test_negative_cycle_detection_positive: OK (cycle={cycle}, poids={total})")


def test_negative_cycle_detection_negative():
    """Graphe avec coût négatif mais pas de cycle. Ne doit PAS être flaggué."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5, cost=2)
    g.add_edge(1, 2, 5, cost=3)
    g.add_edge(2, 3, 5, cost=-4)
    g.add_edge(0, 3, 5, cost=10)
    has, _ = detect_negative_cycle_bellman_ford(g)
    assert has is False, "faux positif sur graphe sans cycle"
    print("test_negative_cycle_detection_negative: OK")


def _graph_with_negative_cycle():
    """Petit graphe avec cycle 1 -> 2 -> 3 -> 1 de poids -3 + 1 - 1 = -3.
    Pas d'arcs antiparallèles réels (la structure ne les supporte pas)."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5, cost=1)
    g.add_edge(1, 2, 5, cost=-3)
    g.add_edge(2, 3, 5, cost=1)
    g.add_edge(3, 1, 5, cost=-1)   # ferme le cycle 1->2->3->1
    g.add_edge(3, 4, 5, cost=2)
    return g


def test_mcf_aborts_on_negative_cycle():
    """Si on lance MCF sur un graphe avec cycle négatif → doit avorter."""
    g = _graph_with_negative_cycle()
    cost, flow, flows = min_cost_flow_bellman_ford(g, 0, 4)
    assert cost is None and flow is None and flows is None
    print("test_mcf_aborts_on_negative_cycle: OK")


def test_dfs_cycle_fallback():
    """Le fallback DFS doit aussi trouver un cycle négatif sur le même graphe."""
    g = _graph_with_negative_cycle()
    has, cycle = detect_negative_cycle_dfs(g)
    assert has, "DFS aurait dû trouver le cycle 1->2->3->1"
    print(f"test_dfs_cycle_fallback: OK (cycle={cycle})")


def test_stress_10_nodes():
    """Stress test sur 10 noeuds — vérifie surtout qu'on ne plante pas et que
    BF et Dijkstra restent cohérents."""
    edges = [
        (0, 1, 10, 2), (0, 2, 8, 4), (1, 3, 6, 3), (2, 3, 4, 1),
        (1, 4, 3, 2), (3, 5, 7, 2), (4, 5, 5, 1), (4, 6, 4, 3),
        (5, 7, 6, 2), (6, 7, 3, 4), (7, 8, 8, 1), (5, 8, 5, 5),
        (6, 8, 4, 2), (8, 9, 12, 1),
    ]

    g_ff = ResidualGraph()
    for u, v, cap, _w in edges:
        g_ff.add_edge(u, v, cap)
    mf, _, _, _, _ = ford_fulkerson(g_ff, 0, 9)
    assert mf > 0

    # MCF pour pousser 5 unités avec les deux algos
    g_bf = ResidualGraph()
    for u, v, cap, w in edges:
        g_bf.add_edge(u, v, cap, cost=w)
    g_dj = ResidualGraph()
    for u, v, cap, w in edges:
        g_dj.add_edge(u, v, cap, cost=w)

    c_bf, f_bf, _ = min_cost_flow_bellman_ford(g_bf, 0, 9, max_flow=5)
    c_dj, f_dj, _ = min_cost_flow_dijkstra(g_dj, 0, 9, max_flow=5)

    assert f_bf == 5 and f_dj == 5
    assert c_bf == c_dj, f"BF={c_bf}, DJ={c_dj}"
    print(f"test_stress_10_nodes: OK (max_flow={mf}, mcf_cost_for_5={c_bf})")


# ---------- runner ----------

ALL_TESTS = [
    test_simple_max_flow,
    test_max_flow_conservation,
    test_ford_fulkerson_known_value,
    test_mcf_bellman_ford_negative_cost,
    test_mcf_dijkstra_johnson_matches_bf,
    test_mcf_dijkstra_on_negative_arc,
    test_negative_cycle_detection_positive,
    test_negative_cycle_detection_negative,
    test_mcf_aborts_on_negative_cycle,
    test_dfs_cycle_fallback,
    test_stress_10_nodes,
]


def run_all():
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"{t.__name__}: ÉCHEC — {e}")
        except Exception as e:
            failed += 1
            print(f"{t.__name__}: EXCEPTION — {type(e).__name__}: {e}")
    print()
    if failed == 0:
        print(f"Tous les tests OK ({len(ALL_TESTS)}/{len(ALL_TESTS)}).")
    else:
        print(f"{failed}/{len(ALL_TESTS)} tests ont échoué.")
    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(0 if failed == 0 else 1)
