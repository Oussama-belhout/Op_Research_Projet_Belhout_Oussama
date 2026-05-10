"""
src/ford_fulkerson.py
Algorithme de flot maximum, version Edmonds-Karp (Ford-Fulkerson + BFS).

J'utilise BFS plutôt que DFS parce que ça garantit qu'on prend le plus court
chemin augmentant en NOMBRE D'ARCS, et c'est ce qui donne la borne polynomiale
— sinon avec DFS on peut tomber sur des cas pathologiques où le nombre
d'augmentations dépend de la valeur du flot (et donc des capacités), ce qui n'est
pas polynomial.
"""

from collections import deque
from typing import Tuple, Dict, Set, List

from src.graph import ResidualGraph


def _bfs_augmenting_path(graph: ResidualGraph, source: int, sink: int) -> List[int]:
    """BFS pour trouver le PLUS COURT chemin augmentant (en nombre d'arcs).

    Renvoie la liste des noeuds du chemin [source, ..., sink], ou [] si pas de chemin.
    """
    if source == sink:
        return [source]

    parent: Dict[int, int] = {source: source}  # source pointe vers elle-même comme sentinelle
    queue = deque([source])

    while queue:
        u = queue.popleft()
        # on regarde tous les voisins de u dans le résiduel
        if u not in graph.graph:
            continue
        for v in graph.graph[u]:
            if v in parent:
                continue
            if graph.residual_capacity(u, v) <= 0:
                continue
            parent[v] = u
            if v == sink:
                # reconstruire le chemin en remontant les parents
                path = [v]
                while path[-1] != source:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(v)

    return []  # plus de chemin augmentant


def _bfs_reachable(graph: ResidualGraph, source: int) -> Set[int]:
    """Ensemble des noeuds atteignables depuis source dans le résiduel.
    Utilisé après convergence pour calculer la coupe min."""
    visited = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u not in graph.graph:
            continue
        for v in graph.graph[u]:
            if v not in visited and graph.residual_capacity(u, v) > 0:
                visited.add(v)
                queue.append(v)
    return visited


def ford_fulkerson(
    graph: ResidualGraph,
    source: int,
    sink: int,
    verbose: bool = False,
) -> Tuple[int, Dict[Tuple[int, int], int], Set[int], Set[int], List[Tuple[int, int]]]:
    """Implémentation de Ford-Fulkerson avec BFS — donc en pratique Edmonds-Karp.

    Renvoie un tuple (max_flow, flots_par_arc, S, T, arcs_de_la_coupe):
      - max_flow: valeur du flot max
      - flots_par_arc: dict {(u,v): flow} sur les arcs réels uniquement
      - S, T: les deux côtés de la coupe min
      - arcs_de_la_coupe: liste des arcs (u,v) traversant la coupe (forcément saturés)

    La coupe min est calculée à la fin par un BFS sur le graphe résiduel depuis
    la source — cette partie m'a pris du temps à bien capter. L'idée: une fois
    qu'il n'y a plus de chemin augmentant, l'ensemble des noeuds atteignables
    depuis s dans le résiduel est S, le reste est T, et la coupe est constituée
    des arcs réels (u,v) avec u∈S, v∈T qui sont SATURÉS — sinon ils auraient
    permis d'augmenter encore (contradiction).



    NB: on ne fait PAS reset_flows() automatiquement, au cas où on voudrait
    reprendre depuis un état existant. Si nécessaire, faire graph.reset_flows()
    avant l'appel.
    """
    iteration = 0
    max_flow = 0

    while True:
        path = _bfs_augmenting_path(graph, source, sink)
        if not path:
            break  # plus de chemin → on a convergé

        # bottleneck = min des capacités résiduelles le long du chemin
        bottleneck = min(
            graph.residual_capacity(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

        graph.augment_path(path, bottleneck)
        max_flow += bottleneck
        iteration += 1

        if verbose:
            print(f"  [FF iter {iteration}] chemin = "
                  f"{' -> '.join(map(str, path))}, "
                  f"bottleneck = {bottleneck}, flot total = {max_flow}")

    # construction du dict {(u,v): flow} sur les arcs réels uniquement
    flows: Dict[Tuple[int, int], int] = {}
    for u, v, attrs in graph.get_real_arcs():
        flows[(u, v)] = attrs['flow']

    # coupe min: BFS sur le résiduel depuis source
    S = _bfs_reachable(graph, source)
    T = graph.nodes - S
    cut_arcs: List[Tuple[int, int]] = []
    for u in S:
        if u not in graph.graph:
            continue
        for v in graph.graph[u]:
            if v in T and graph.graph[u][v]['capacity'] > 0:
                # arc réel qui traverse la coupe — il doit être saturé sinon
                # le BFS l'aurait franchi
                cut_arcs.append((u, v))

    if verbose:
        print(f"  [FF] flot max = {max_flow}, "
              f"#augmentations = {iteration}")
        print(f"  [FF] coupe min: |S|={len(S)}, |T|={len(T)}")
        print(f"  [FF] arcs de la coupe: {cut_arcs}")

    return max_flow, flows, S, T, cut_arcs
