"""
src/negative_cycle.py
Détection de cycles négatifs.

Méthode principale: Bellman-Ford. Après V-1 relaxations on a les plus courts
chemins SI il n'y a pas de cycle négatif. Si une V-ième relaxation améliore
encore une distance, c'est qu'il y a un cycle négatif quelque part.

Pour détecter les cycles même non atteignables depuis une source donnée, on
utilise l'astuce de la "source virtuelle": on initialise toutes les distances
à 0 (ce qui revient à ajouter un noeud relié à tous les autres avec coût 0).

Méthode de fallback: DFS sur le graphe original pour trouver UN cycle quelconque
puis vérifier son poids. Moins puissant en théorie mais utile comme double-check.
"""

from typing import List, Optional, Tuple

from src.graph import ResidualGraph


def detect_negative_cycle_bellman_ford(
    graph: ResidualGraph,
    verbose: bool = False,
) -> Tuple[bool, List[int]]:
    """Détecte un cycle négatif via Bellman-Ford avec source virtuelle.

    Renvoie (a_un_cycle, cycle_extrait_si_oui).
    Le cycle est renvoyé comme une liste [v0, v1, ..., vk, v0] (donc le premier
    et dernier élément sont identiques pour rendre les arcs explicites).

    NB: on ne regarde que les arcs RÉELS (capacity > 0). Pour appliquer la
    détection sur le résiduel pendant un MCF, c'est un autre besoin et on n'en
    a pas vraiment besoin parce que: si le graphe original n'a pas de cycle
    négatif, alors aucun résiduel construit pendant MCF n'en aura non plus
    (théorème classique — c'est l'invariant qu'on prouve dans le rapport).
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return False, []

    # source virtuelle implicite: dist[v] = 0 partout au départ
    # (équivaut à un noeud s* relié à tous avec coût 0)
    dist = {v: 0 for v in nodes}
    parent: dict = {v: None for v in nodes}

    # collecter les arcs RÉELS une fois pour toutes (plus rapide que reparcourir
    # le dict à chaque itération)
    edges = []
    for u in graph.graph:
        for v in graph.graph[u]:
            if graph.graph[u][v]['capacity'] > 0:  # arc réel uniquement
                edges.append((u, v, graph.graph[u][v]['cost']))

    # n-1 itérations de relaxation
    for i in range(n - 1):
        updated = False
        for (u, v, w) in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            # convergé tôt → pas de cycle négatif possible
            if verbose:
                print(f"  [BF] convergé après {i+1} itérations, pas de cycle négatif")
            return False, []

    # n-ième relaxation: si ça change encore, c'est qu'il y a un cycle négatif
    victim: Optional[int] = None
    for (u, v, w) in edges:
        if dist[u] + w < dist[v]:
            victim = v
            parent[v] = u
            break

    if victim is None:
        return False, []

    # extraction du cycle:
    # on remonte n fois via parent pour être sûr d'atterrir DANS le cycle (pas
    # dans la "queue" qui mène au cycle). C'est une astuce classique vue en cours,
    # ça marche parce qu'après n pas on a forcément bouclé une fois.
    for _ in range(n):
        if victim is None:
            return False, []
        victim = parent[victim]
    if victim is None:
        return False, []

    # maintenant on suit parent depuis victim jusqu'à le retrouver — ça nous donne
    # le cycle dans l'ordre INVERSE des arcs. Donc on reverse à la fin.
    cycle = [victim]
    cur = parent[victim]
    safety = 0
    while cur != victim:
        cycle.append(cur)
        cur = parent[cur]
        safety += 1
        if safety > n + 1:
            # garde-fou — ne devrait pas arriver
            break
    cycle.append(victim)
    cycle.reverse()

    if verbose:
        print(f"  [BF] CYCLE NÉGATIF détecté: {' -> '.join(map(str, cycle))}")

    return True, cycle


def detect_negative_cycle_dfs(
    graph: ResidualGraph,
    verbose: bool = False,
) -> Tuple[bool, List[int]]:
    """Fallback: DFS sur le graphe pour trouver UN cycle, puis check si négatif.

    Honnêtement c'est moins puissant que BF: on trouve le PREMIER cycle rencontré
    et on regarde si il est négatif. Si pas, on continue à chercher d'autres cycles.
    Sur des graphes avec beaucoup de cycles non négatifs ça peut rater, mais
    pour notre projet ça sert juste de double-check sur des petits graphes.

    NB: la détection de cycle négatif "robuste" via DFS (sans Bellman-Ford) est
    en fait un problème non trivial — la méthode "propre" reste BF. Cette fonction
    est là parce que le sujet demandait un fallback.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph.nodes}

    def dfs(start: int) -> Optional[List[int]]:
        # DFS itérative pour éviter les soucis de récursion sur les gros graphes
        # — on garde une pile de (noeud, itérateur sur ses voisins)
        stack: List[Tuple[int, list, int]] = []
        path: List[int] = []
        # on garde aussi un set pour tester l'appartenance à path en O(1)
        on_path = set()

        if start not in graph.graph:
            return None

        color[start] = GRAY
        path.append(start)
        on_path.add(start)
        stack.append((start, [v for v in graph.graph[start]
                              if graph.graph[start][v]['capacity'] > 0], 0))

        while stack:
            u, neighbors, idx = stack[-1]
            if idx >= len(neighbors):
                # fini d'explorer u
                color[u] = BLACK
                path.pop()
                on_path.discard(u)
                stack.pop()
                continue
            v = neighbors[idx]
            stack[-1] = (u, neighbors, idx + 1)
            if color[v] == GRAY and v in on_path:
                # cycle trouvé: portion de path depuis v
                idx_v = path.index(v)
                return path[idx_v:] + [v]
            if color[v] == WHITE:
                color[v] = GRAY
                path.append(v)
                on_path.add(v)
                next_neighbors = [w for w in graph.graph.get(v, {})
                                  if graph.graph[v][w]['capacity'] > 0]
                stack.append((v, next_neighbors, 0))
        return None

    def cycle_weight(cycle: List[int]) -> int:
        total = 0
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            if u in graph.graph and v in graph.graph[u]:
                total += graph.graph[u][v]['cost']
        return total

    for start in list(graph.nodes):
        if color[start] == WHITE:
            cycle = dfs(start)
            if cycle is not None:
                w = cycle_weight(cycle)
                if w < 0:
                    if verbose:
                        print(f"  [DFS] cycle négatif: {cycle} (poids={w})")
                    return True, cycle
                else:
                    if verbose:
                        print(f"  [DFS] cycle trouvé mais pas négatif "
                              f"(poids={w}): {cycle}")
                    # on ne s'arrête pas — on cherche ailleurs au cas où il y
                    # aurait un cycle négatif dans une autre composante

    return False, []
