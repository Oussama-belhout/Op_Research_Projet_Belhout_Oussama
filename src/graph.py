"""
src/graph.py
Structure de graphe résiduel pour les algos de flot.

Choix d'implémentation: dict de dict, avec graph[u][v] = {capacity, flow, cost}.
Pour chaque arc réel (u,v), on crée AUSSI un arc inverse (v,u) avec capacity=0
et cost=-w. Comme ça la capacité résiduelle est toujours capacity - flow,
peu importe si l'arc est "réel" ou "inverse". J'ai un peu galéré au début à
comprendre pourquoi on stockait le flot négatif sur l'arc inverse, mais en
fait c'est juste pour que la formule capacity - flow marche dans les deux sens.
"""

from typing import Dict, List, Tuple, Set


class ResidualGraph:
    """Graphe résiduel utilisé par tous les algos du projet (FF, MCF, etc.).

    Le stockage interne, c'est un dict de dict:
        self.graph[u][v] = {'capacity': ..., 'flow': ..., 'cost': ...}

    À chaque appel à add_edge(u, v, c, w), on crée DEUX entrées:
      - graph[u][v] avec capacity=c, flow=0, cost=w        ← arc "réel"
      - graph[v][u] avec capacity=0, flow=0, cost=-w       ← arc inverse résiduel

    L'avantage: capacité résiduelle dans n'importe quelle direction = capacity - flow.
    Quand on pousse delta sur (u,v), on incrémente flow(u,v) de +delta et flow(v,u)
    de -delta. Du coup pour l'inverse: capacity(v,u) - flow(v,u) = 0 - (-delta) = delta,
    ce qui correspond exactement au flot qu'on peut "annuler". Élégant.

    Limitation connue: si on a un arc réel (u,v) ET un arc réel (v,u), ils vont se
    marcher sur les pieds dans cette structure. On lève une exception dans ce cas et
    on note dans le rapport qu'il faudrait dédoubler avec un noeud intermédiaire si
    on en avait besoin (typiquement on n'en a pas en MCF standard).
    """

    def __init__(self) -> None:
        self.graph: Dict[int, Dict[int, dict]] = {}
        self.nodes: Set[int] = set()

    # ---------- construction ----------

    def add_edge(self, u: int, v: int, capacity: int, cost: int = 0) -> None:
        """Ajoute un arc (u,v) avec capacité et coût (coût optionnel pour Ford-Fulkerson pur).

        Si l'arc (u,v) existe déjà comme arc réel, on additionne les capacités
        (cas d'arcs parallèles). Si (v,u) existe déjà comme arc réel, on refuse —
        notre structure ne gère pas les arcs antiparallèles tels quels.
        """
        if u == v:
            # boucle, ça n'a pas de sens dans un problème de flot
            return
        if capacity < 0:
            raise ValueError(f"capacité négative interdite: {capacity}")

        if u not in self.graph:
            self.graph[u] = {}
        if v not in self.graph:
            self.graph[v] = {}

        # cas arc parallèle: même direction, déjà réel → on cumule la capacité
        if v in self.graph[u] and self.graph[u][v]['capacity'] > 0:
            self.graph[u][v]['capacity'] += capacity
            # NB: si les coûts diffèrent on garde le premier — pas idéal mais ce
            # cas ne devrait pas arriver dans nos jeux de tests. À voir si besoin.
            return

        # cas arc antiparallèle réel: pas supporté (workaround = noeud intermédiaire)
        if u in self.graph[v] and self.graph[v][u]['capacity'] > 0:
            raise ValueError(
                f"Arc antiparallèle réel {v}->{u} déjà présent. "
                f"Pour gérer ce cas il faudrait dédoubler un des deux arcs avec "
                f"un noeud intermédiaire."
            )

        # ajout normal
        self.graph[u][v] = {'capacity': capacity, 'flow': 0, 'cost': cost}
        # arc inverse: capacité 0 (au début rien à annuler), coût opposé
        if u not in self.graph[v]:
            self.graph[v][u] = {'capacity': 0, 'flow': 0, 'cost': -cost}

        self.nodes.add(u)
        self.nodes.add(v)

    # ---------- accès / requêtes ----------

    def residual_capacity(self, u: int, v: int) -> int:
        """Capacité résiduelle de (u,v).

        Pour un arc réel (u,v): capacity - flow.
        Pour un arc inverse: 0 - (-flot_arc_réel) = flot_arc_réel.
        Dans les deux cas la formule est la même grâce au stockage symétrique du flot.
        """
        if u not in self.graph or v not in self.graph[u]:
            return 0
        return self.graph[u][v]['capacity'] - self.graph[u][v]['flow']

    def neighbors_with_capacity(self, u: int) -> List[int]:
        """Voisins de u accessibles dans le résiduel (capacité résiduelle > 0).
        Petit helper utilisé par BFS et Dijkstra. Pas indispensable mais ça rend le
        code des algos plus lisible."""
        if u not in self.graph:
            return []
        return [v for v in self.graph[u] if self.residual_capacity(u, v) > 0]

    def get_real_arcs(self) -> List[Tuple[int, int, dict]]:
        """Renvoie seulement les arcs "réels" (capacity > 0).

        Sert pour afficher les flots à la fin sans inclure les arcs inverses
        qui ne sont qu'un artefact de notre stockage."""
        arcs = []
        for u in self.graph:
            for v in self.graph[u]:
                if self.graph[u][v]['capacity'] > 0:
                    arcs.append((u, v, self.graph[u][v]))
        return arcs

    # ---------- modifications de flot ----------

    def augment_path(self, path: List[int], delta: int) -> None:
        """Pousse delta unités de flot le long du chemin [s, ..., t].

        On met à jour le flot dans les DEUX directions à la fois (cf. l'astuce
        de stockage symétrique du flot)."""
        if delta <= 0:
            return  # garde-fou — on n'augmente jamais avec 0 ou négatif
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['flow'] += delta
            self.graph[v][u]['flow'] -= delta

    def reset_flows(self) -> None:
        """Remet tous les flots à 0. Utile entre deux runs sur le même graphe."""
        for u in self.graph:
            for v in self.graph[u]:
                self.graph[u][v]['flow'] = 0

    # ---------- utilitaires divers ----------

    def copy(self) -> 'ResidualGraph':
        """Copie indépendante. Pratique pour relancer un algo sans toucher l'original."""
        new_g = ResidualGraph()
        new_g.nodes = set(self.nodes)
        for u in self.graph:
            new_g.graph[u] = {}
            for v in self.graph[u]:
                # copie du dict des attributs de l'arc
                new_g.graph[u][v] = dict(self.graph[u][v])
        return new_g

    def __repr__(self) -> str:
        # n'affiche que les arcs réels — l'inverse est juste de la plomberie interne
        lines = [f"ResidualGraph({len(self.nodes)} noeuds, "
                 f"{len(self.get_real_arcs())} arcs):"]
        for u in sorted(self.graph.keys()):
            for v in sorted(self.graph[u].keys()):
                a = self.graph[u][v]
                if a['capacity'] > 0:
                    lines.append(f"  {u} -> {v}: cap={a['capacity']}, "
                                 f"flow={a['flow']}, cost={a['cost']}")
        return "\n".join(lines)
