# Projet d'OR — Algorithmes de flot

Implémentation pédagogique de :

- **Ford-Fulkerson (Edmonds-Karp)** pour le flot maximum + coupe minimum.
- **Flot de coût minimum** (MCF) par augmentations successives, en deux versions :
  - avec Bellman-Ford à chaque itération,
  - avec Dijkstra + renormalisation de Johnson.
- **Détection de cycles négatifs** (Bellman-Ford + fallback DFS).

Aucune bibliothèque externe — uniquement Python 3 standard, `collections` et
`heapq`.

## Arborescence

```
OR_project/
├── src/
│   ├── graph.py             # structure de graphe résiduel (dict de dict)
│   ├── ford_fulkerson.py    # flot max via BFS
│   ├── min_cost_flow.py     # MCF (deux variantes)
│   └── negative_cycle.py    # détection de cycle négatif
├── tests/
│   └── test_all.py
├── main.py                  # exécution des exemples + affichage
├── rapport.tex              # rapport détaillé en LaTeX (à compiler sur Overleaf)
├── rapport.md               # version Markdown de secours (même contenu)
└── README.md
```

## Rapport

Le rapport principal est `rapport.tex`, à compiler sur **Overleaf**
(ou avec `pdflatex` en local). Il suffit de :

1. Créer un nouveau projet sur Overleaf.
2. Y coller le contenu de `rapport.tex`.
3. Compiler (deux passes pour la table des matières).

Le rapport utilise uniquement des paquets standards (`babel`, `amsmath`,
`amsthm`, `tikz`, `algorithm`, `algpseudocode`, `listings`, `booktabs`,
`tcolorbox`, `hyperref`) tous présents par défaut sur Overleaf.

## Lancer le projet

Depuis la racine du projet :

```
python main.py
```

Ça affiche les résultats pour chaque algorithme dans le format demandé.

## Lancer les tests

```
python tests/test_all.py
```

Tous les tests doivent passer (`Tous les tests OK`). Le test
`test_mcf_dijkstra_johnson_matches_bf` est le plus important : il compare les
deux versions du MCF sur le même graphe et vérifie qu'elles donnent
exactement le même coût et le même flot.

## Mode verbose

Tous les algorithmes acceptent un paramètre `verbose=True` qui imprime
chaque chemin augmentant trouvé et le bottleneck/coût correspondant. Utile
pour le débogage et pour comprendre le déroulé.

Exemple :

```python
from src.graph import ResidualGraph
from src.ford_fulkerson import ford_fulkerson

g = ResidualGraph()
g.add_edge(0, 1, 3)
g.add_edge(0, 2, 2)
g.add_edge(1, 3, 2)
g.add_edge(2, 3, 3)
g.add_edge(1, 2, 1)
ford_fulkerson(g, 0, 3, verbose=True)
```

## Contenu du rapport

Le rapport couvre :

- justification des choix de structure de données (dict de dict, astuce du flot symétrique) ;
- pseudo-code détaillé de chaque algorithme ;
- analyse de complexité (Edmonds--Karp en O(VE²), MCF-BF en O(F·VE), MCF-Johnson en O(VE + F(V+E)log V)) ;
- preuves : maintien de `w'(u,v) ≥ 0` après renormalisation de Johnson (sur les arcs existants ET sur les nouveaux arcs inverses) ;
- comparaison qualitative des deux approches MCF.
