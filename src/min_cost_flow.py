"""
src/min_cost_flow.py
Flot à coût minimum par augmentations successives.

Deux variantes:
  1) Bellman-Ford à chaque itération — gère naturellement les coûts négatifs sur
     les arcs, mais O(V*E) par itération donc lent.
  2) Dijkstra + reweighting de Johnson — Bellman-Ford UNE FOIS pour initialiser
     les potentiels, puis Dijkstra (rapide grâce au tas binaire de heapq) pour
     toutes les augmentations suivantes. Beaucoup plus rapide en pratique.

Avant de lancer une augmentation on vérifie qu'il n'y a pas de cycle négatif
dans le graphe original — sinon le problème est mal posé (coût non borné
inférieurement) et on avorte proprement avec un message.
"""

import heapq
from typing import Tuple, Dict, Optional

from src.graph import ResidualGraph
from src.negative_cycle import detect_negative_cycle_bellman_ford


INF = float('inf')


# =================================================================
# 1) Variante Bellman-Ford (Successive Shortest Paths avec BF)
# =================================================================

def _bellman_ford_residual(
    graph: ResidualGraph,
    source: int,
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Bellman-Ford sur le graphe résiduel: renvoie (dist, parent) depuis source.

    Important: on considère TOUS les arcs avec capacité résiduelle > 0, donc à
    la fois les arcs réels ET les arcs inverses du résiduel. Les arcs inverses
    ont des coûts négatifs (-w pour chaque arc réel de coût w utilisé), c'est
    précisément la raison pour laquelle Dijkstra direct ne marche pas et
    pourquoi on a besoin de Bellman-Ford ici.
    """
    dist: Dict[int, float] = {v: INF for v in graph.nodes}
    parent: Dict[int, Optional[int]] = {v: None for v in graph.nodes}
    dist[source] = 0

    n = len(graph.nodes)

    for _ in range(n - 1):
        updated = False
        for u in graph.graph:
            if dist[u] == INF:
                continue  # rien à propager depuis ce noeud pour le moment
            for v in graph.graph[u]:
                if graph.residual_capacity(u, v) <= 0:
                    continue
                w = graph.graph[u][v]['cost']
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    updated = True
        if not updated:
            break  # convergence anticipée — pas la peine de continuer

    return dist, parent


def min_cost_flow_bellman_ford(
    graph: ResidualGraph,
    source: int,
    sink: int,
    max_flow: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Optional[int], Optional[int], Optional[Dict[Tuple[int, int], int]]]:
    """MCF par augmentations successives en utilisant Bellman-Ford à chaque tour.

    Paramètres:
      - graph, source, sink: comme d'hab
      - max_flow: si fourni, on s'arrête dès qu'on a poussé ce volume. Sinon on
        pousse autant que possible (jusqu'à plus aucun chemin).
      - verbose: affichage du chemin et coût à chaque itération

    Renvoie (cost_total, flow_total, dict_flots_par_arc).
    Si cycle négatif détecté → renvoie (None, None, None) après affichage d'un
    message d'avertissement. Le problème de MCF avec cycle négatif est mal posé
    (on pourrait diminuer le coût indéfiniment en faisant tourner du flot autour
    du cycle), donc on refuse de tourner dans ce cas.

    Complexité: O(F · V·E) avec F = volume de flot total. Quand F est grand
    (capacités élevées), c'est la version Dijkstra+Johnson qu'il faut utiliser.
    """
    # 1) sanity check: pas de cycle négatif dans l'original
    has_neg, cycle = detect_negative_cycle_bellman_ford(graph)
    if has_neg:
        print(f"[MCF-BF] AVORT: cycle négatif détecté dans le graphe original: "
              f"{' -> '.join(map(str, cycle))}")
        return None, None, None

    total_cost = 0
    total_flow = 0
    iteration = 0

    while True:
        if max_flow is not None and total_flow >= max_flow:
            break

        dist, parent = _bellman_ford_residual(graph, source)
        if dist[sink] == INF:
            break  # plus aucun chemin du source au sink dans le résiduel

        # reconstruction du chemin par remontée des parents
        path = [sink]
        while path[-1] != source:
            p = parent[path[-1]]
            if p is None:
                # ne devrait pas arriver puisque dist[sink] != INF
                break
            path.append(p)
        path.reverse()

        # bottleneck = min des capacités résiduelles
        bottleneck = min(
            graph.residual_capacity(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )
        if max_flow is not None:
            bottleneck = min(bottleneck, max_flow - total_flow)

        graph.augment_path(path, bottleneck)
        path_cost_per_unit = dist[sink]
        total_cost += int(path_cost_per_unit * bottleneck)
        total_flow += bottleneck
        iteration += 1

        if verbose:
            print(f"  [MCF-BF iter {iteration}] chemin = "
                  f"{' -> '.join(map(str, path))}, "
                  f"bottleneck = {bottleneck}, "
                  f"coût unitaire = {path_cost_per_unit}, "
                  f"cumul flot = {total_flow}, cumul coût = {total_cost}")

    flows: Dict[Tuple[int, int], int] = {}
    for u, v, attrs in graph.get_real_arcs():
        flows[(u, v)] = attrs['flow']

    return total_cost, total_flow, flows


# =================================================================
# 2) Variante Dijkstra + reweighting de Johnson
# =================================================================

def _dijkstra_reweighted(
    graph: ResidualGraph,
    source: int,
    potential: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Dijkstra sur le graphe résiduel avec coûts reweightés via les potentiels h.

    Coût reweighté: w'(u,v) = w(u,v) + h[u] - h[v].
    Si h est cohérent (= distances depuis source dans le graphe précédent),
    alors w'(u,v) >= 0 pour tout arc avec capacité résiduelle > 0 — preuve dans
    le rapport. Du coup Dijkstra (heap) marche, et c'est rapide.
    """
    dist: Dict[int, float] = {v: INF for v in graph.nodes}
    parent: Dict[int, Optional[int]] = {v: None for v in graph.nodes}
    dist[source] = 0

    pq: list = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue  # entrée périmée dans le tas, on ignore
        if u not in graph.graph:
            continue
        for v in graph.graph[u]:
            if graph.residual_capacity(u, v) <= 0:
                continue
            # edge case: si un noeud n'a pas de potentiel fini (pas atteignable
            # depuis source dans une itération précédente), on le skip — sinon
            # on aurait inf - inf = NaN. C'est ce qui m'a fait galérer pendant
            # une heure, en fait.
            if potential[u] == INF or potential[v] == INF:
                continue
            w = graph.graph[u][v]['cost'] + potential[u] - potential[v]
            if w < 0:
                # ne devrait jamais arriver si Johnson est correct — sinon c'est
                # un bug dans la mise à jour des potentiels
                raise RuntimeError(
                    f"poids reweighté négatif w'({u},{v}) = {w} — bug Johnson"
                )
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent


def min_cost_flow_dijkstra(
    graph: ResidualGraph,
    source: int,
    sink: int,
    max_flow: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Optional[int], Optional[int], Optional[Dict[Tuple[int, int], int]]]:
    """MCF avec Dijkstra + reweighting de Johnson.

    Étape 1: Bellman-Ford UNE seule fois pour obtenir les potentiels initiaux h
             (= plus courtes distances depuis source dans le graphe original).
    Étape 2: à chaque itération, Dijkstra avec heapq sur les coûts reweightés
             w'(u,v) = w(u,v) + h[u] - h[v].
    Étape 3: après chaque Dijkstra, mise à jour h_new[v] = h[v] + dist[v] (les
             distances Dijkstra étant celles sur les coûts reweightés). Ça
             maintient l'invariant w' >= 0 pour la prochaine itération — preuve
             dans le rapport.

    Pour récupérer le coût RÉEL d'un chemin (pas le reweighté):
        cost_réel = dist_reweighted[sink] + h[sink] - h[source]
    Ça vient du télescopage: la somme des (h[u] - h[v]) le long du chemin se
    simplifie en h[s] - h[t]. J'ai mis 10 minutes à m'en convaincre la première
    fois.

    Complexité: O(V·E) initial + O(F · (V+E) log V) en augmentations. C'est
    largement meilleur que la version BF dès que F est non négligeable.

    Limitation à signaler: si certains noeuds sont inaccessibles depuis source
    et le DEVIENNENT plus tard via des arcs inverses, leur potentiel reste à
    INF et ils ne seront jamais visités. Pour nos tests c'est OK parce qu'on
    suppose les graphes "bien posés" (tout noeud accessible depuis s).
    """
    # 1) sanity check
    has_neg, cycle = detect_negative_cycle_bellman_ford(graph)
    if has_neg:
        print(f"[MCF-DJ] AVORT: cycle négatif détecté dans le graphe original: "
              f"{' -> '.join(map(str, cycle))}")
        return None, None, None

    # 2) potentiels initiaux via BF
    potential, _ = _bellman_ford_residual(graph, source)

    total_cost = 0
    total_flow = 0
    iteration = 0

    while True:
        if max_flow is not None and total_flow >= max_flow:
            break

        dist, parent = _dijkstra_reweighted(graph, source, potential)
        if dist[sink] == INF:
            break  # plus de chemin

        # reconstruction du chemin
        path = [sink]
        while path[-1] != source:
            p = parent[path[-1]]
            if p is None:
                break
            path.append(p)
        path.reverse()

        # bottleneck
        bottleneck = min(
            graph.residual_capacity(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )
        if max_flow is not None:
            bottleneck = min(bottleneck, max_flow - total_flow)

        # coût RÉEL par unité (formule du télescopage)
        # NB: on calcule AVANT d'augmenter et AVANT de mettre à jour les potentiels
        real_cost_per_unit = dist[sink] + potential[sink] - potential[source]

        graph.augment_path(path, bottleneck)
        total_cost += int(real_cost_per_unit * bottleneck)
        total_flow += bottleneck
        iteration += 1

        if verbose:
            print(f"  [MCF-DJ iter {iteration}] chemin = "
                  f"{' -> '.join(map(str, path))}, "
                  f"bottleneck = {bottleneck}, "
                  f"coût unitaire (réel) = {real_cost_per_unit}, "
                  f"cumul flot = {total_flow}, cumul coût = {total_cost}")

        # mise à jour des potentiels: h[v] += dist[v] pour les noeuds atteints
        for v in graph.nodes:
            if dist[v] < INF and potential[v] < INF:
                potential[v] = potential[v] + dist[v]
            # sinon on laisse comme avant — voir docstring sur la limitation

    flows: Dict[Tuple[int, int], int] = {}
    for u, v, attrs in graph.get_real_arcs():
        flows[(u, v)] = attrs['flow']

    return total_cost, total_flow, flows
