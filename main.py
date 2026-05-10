"""
main.py
Point d'entrée du projet.

Construit quelques graphes d'exemple et lance:
  1) Ford-Fulkerson (Edmonds-Karp) pour le flot maximum
  2) MCF par augmentations successives avec Bellman-Ford
  3) MCF avec Dijkstra + reweighting de Johnson
  4) Détection de cycle négatif sur deux graphes (un sans, un avec cycle)

Format de sortie conforme au sujet.
"""

from src.graph import ResidualGraph
from src.ford_fulkerson import ford_fulkerson
from src.min_cost_flow import min_cost_flow_bellman_ford, min_cost_flow_dijkstra
from src.negative_cycle import detect_negative_cycle_bellman_ford


# ---------- petits graphes d'exemple ----------

def example_max_flow() -> ResidualGraph:
    """4 noeuds, exemple typique vu en TD."""
    g = ResidualGraph()
    g.add_edge(0, 1, 3)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 2)
    g.add_edge(2, 3, 3)
    return g


def example_mcf_negative() -> ResidualGraph:
    """Graphe avec un coût négatif sur un arc — montre que BF gère ça bien."""
    g = ResidualGraph()
    g.add_edge(0, 1, 4, cost=2)
    g.add_edge(0, 2, 3, cost=1)
    g.add_edge(1, 2, 2, cost=-1)   # ← arc à coût négatif
    g.add_edge(1, 3, 3, cost=4)
    g.add_edge(2, 3, 4, cost=2)
    return g


def example_mcf_positive() -> ResidualGraph:
    """Graphe à coûts positifs — Johnson devrait reweighter en gardant des coûts >= 0
    dès le début et fonctionner sans souci."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5, cost=3)
    g.add_edge(0, 2, 4, cost=2)
    g.add_edge(1, 2, 2, cost=1)
    g.add_edge(1, 3, 3, cost=2)
    g.add_edge(2, 3, 5, cost=4)
    return g


def example_no_neg_cycle() -> ResidualGraph:
    """Coût négatif présent mais pas de cycle. Pour vérifier qu'on ne flag pas
    par erreur."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5, cost=2)
    g.add_edge(1, 2, 5, cost=3)
    g.add_edge(2, 3, 5, cost=-4)
    g.add_edge(0, 3, 5, cost=10)
    return g


def example_neg_cycle() -> ResidualGraph:
    """Graphe avec cycle négatif: 2 -> 3 -> 4 -> 2 de poids -3 + 2 - 2 = -3."""
    g = ResidualGraph()
    g.add_edge(0, 1, 5, cost=1)
    g.add_edge(1, 2, 5, cost=1)
    g.add_edge(2, 3, 5, cost=-3)
    g.add_edge(3, 4, 5, cost=2)
    g.add_edge(4, 2, 5, cost=-2)
    g.add_edge(4, 5, 5, cost=1)
    return g


def main() -> None:
    # =============== 1) Ford-Fulkerson ===============
    print("=== FORD-FULKERSON ===")
    g = example_max_flow()
    mf, flows, S, T, cut = ford_fulkerson(g, source=0, sink=3)
    print(f"Max flow value: {mf}")
    print(f"Flow on arcs: {flows}")
    print(f"Min cut S={set(S)} T={set(T)}")
    print(f"Cut arcs: {cut}")

    # =============== 2) MCF — Bellman-Ford ===============
    print()
    print("=== MCF - BELLMAN-FORD ===")
    g = example_mcf_negative()
    cost, flow, flows = min_cost_flow_bellman_ford(g, source=0, sink=3)
    print(f"Min cost: {cost}  | Total flow: {flow}")

    # =============== 3) MCF — Dijkstra + Johnson ===============
    print()
    print("=== MCF - DIJKSTRA + JOHNSON ===")
    g = example_mcf_positive()
    cost, flow, flows = min_cost_flow_dijkstra(g, source=0, sink=3)
    print(f"Min cost: {cost}  | Total flow: {flow}")

    # =============== 4) Détection de cycle négatif ===============
    print()
    print("=== NEGATIVE CYCLE DETECTION ===")
    g1 = example_no_neg_cycle()
    has1, _ = detect_negative_cycle_bellman_ford(g1)
    print(f"Graph 1 (no cycle): {has1}")
    g2 = example_neg_cycle()
    has2, cyc2 = detect_negative_cycle_bellman_ford(g2)
    cycle_str = "->".join(map(str, cyc2)) if cyc2 else ""
    print(f"Graph 2 (has cycle): {has2} — cycle: [{cycle_str}]")


if __name__ == "__main__":
    main()
