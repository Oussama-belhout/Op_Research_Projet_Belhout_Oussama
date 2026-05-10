# Projet de Recherche Opérationnelle — Algorithmes de flot

**Auteur** : *(étudiant M1)*
**Cours** : Recherche Opérationnelle — UNICA M1

---

## Introduction

L'objectif de ce projet est d'implémenter, sans bibliothèque externe spécialisée
(donc pas de `networkx.maximum_flow` ni similaire), trois familles d'algorithmes
sur les réseaux de transport :

1. **Flot maximum** par l'algorithme de Ford-Fulkerson, plus précisément la
   variante Edmonds-Karp qui choisit le plus court chemin augmentant via BFS.
2. **Flot de coût minimum (MCF)** par augmentations successives, en deux versions :
   l'une avec Bellman-Ford, l'autre avec Dijkstra et la renormalisation de Johnson.
3. **Détection de cycles négatifs**, indispensable avant de lancer un MCF (sinon
   le problème est mal posé).

Le code est en Python pur (juste `collections` et `heapq` pour les structures), et
on a fait l'effort de garder un style lisible plutôt que de chasser des
optimisations qui rendraient tout illisible.

---

## 1. Structure du graphe résiduel

C'est probablement le choix le plus structurant du projet, donc on s'y attarde.

### 1.1 Dict de dict vs matrice d'adjacence

Au début on hésitait entre deux représentations :

- **Matrice d'adjacence** : un tableau 2D `M[u][v] = (cap, flow, cost)`. Accès en
  O(1), simple à comprendre, mais on stocke des "cases vides" pour tous les arcs
  qui n'existent pas, ce qui pour un graphe creux (cas typique en MCF) est un
  gaspillage massif de mémoire et fait des O(V) pour parcourir les voisins même
  quand un noeud n'a que 2 ou 3 voisins.
- **Dict de dict** : `graph[u][v] = {capacity, flow, cost}`. Accès en O(1) en
  moyenne (dict Python), parcours des voisins en O(deg(u)), mémoire proportionnelle
  au nombre d'arcs. C'est ce qu'on a retenu.

Pour les graphes qu'on manipule (typiquement |V| dans les dizaines, |E| dans les
centaines), la matrice serait carrément acceptable, mais on a préféré une
structure qui passe à l'échelle si on veut tester sur du plus gros un jour. Et
ça reste simple à manipuler.

### 1.2 Stockage symétrique des arcs résiduels

Le vrai point délicat : pour chaque arc *réel* `(u, v)` qu'on ajoute, on crée
**aussi** une entrée `graph[v][u]` qui représente l'arc *inverse résiduel*.
Cet arc inverse a au départ une capacité **nulle** et un coût opposé.

```
Arc original  : u --[cap=c, flow=f, cost=w]--> v

Stockage interne :
    graph[u][v] = { capacity:  c, flow:  f, cost:  w }   ← arc réel
    graph[v][u] = { capacity:  0, flow: -f, cost: -w }   ← arc inverse résiduel
```

Pourquoi du flot négatif sur l'arc inverse ? Parce qu'on veut que la formule
**`residual_capacity(u, v) = capacity(u, v) - flow(u, v)`** marche dans les
deux sens :

- Sur l'arc réel `(u, v)` :  `res = c - f` (combien on peut encore pousser).
- Sur l'arc inverse `(v, u)` :  `res = 0 - (-f) = f` (combien on peut annuler).

Et quand on augmente le flot de `δ` le long de `(u, v)` :

```
graph[u][v]['flow'] += δ
graph[v][u]['flow'] -= δ
```

Du coup les deux directions restent cohérentes sans avoir à raisonner
séparément sur le résiduel.

Au début on stockait deux structures distinctes (`real_edges` et `residual_edges`)
et c'était insoutenable, on devait toujours faire attention à laquelle on mettait
à jour. Avec le truc du flot symétrique, tout passe par les mêmes deux opérations
et c'est beaucoup plus robuste.

### 1.3 Schéma ASCII

```
                Original :  u  ──(cap=5, cost=2)──>  v   (flot courant = 3)

Stockage interne :
                     cap=5, flow=3, cost=2
                  ┌─────────────────────────────┐
                  │                             │
                  u                             v
                  │                             │
                  └─────────────────────────────┘
                     cap=0, flow=-3, cost=-2

Capacité résiduelle :
    u → v :  5  -  3  =  2     (on peut encore pousser 2 unités)
    v → u :  0  - (-3) =  3    (on peut annuler jusqu'à 3 unités déjà poussées)
```

### 1.4 Limitations connues

Avec cette structure, si on a un arc réel `(u, v)` **et** un arc réel `(v, u)`
(arcs antiparallèles), on a un conflit : l'arc inverse de `(u, v)` doit cohabiter
avec le vrai arc `(v, u)`. La solution standard est de dédoubler un des deux
arcs avec un noeud intermédiaire `x` : `(v, x)` et `(x, u)` au lieu de `(v, u)`.
On lève une `ValueError` si l'utilisateur essaye ce cas, plutôt que de produire
un résultat incorrect en silence.

---

## 2. Ford-Fulkerson (Edmonds-Karp)

### 2.1 Pseudo-code

```
fonction Ford-Fulkerson(G, s, t) :
    tant que vrai :
        chemin = BFS(G, s, t)             # plus court chemin augmentant
        si chemin est vide :
            sortir
        δ = min { residual_capacity(u, v)  pour (u, v) sur le chemin }
        pour chaque arc (u, v) du chemin :
            graph[u][v].flow += δ
            graph[v][u].flow -= δ
        max_flow += δ
    # à ce stade : plus de chemin augmentant → flot max atteint
    S = BFS(G, s)                          # atteignables dans le résiduel
    T = V \ S
    coupe = { (u, v) réel : u ∈ S, v ∈ T }  # tous saturés
    renvoyer (max_flow, S, T, coupe)
```

### 2.2 Justification du choix de BFS

Avec Ford-Fulkerson "tel quel" (n'importe quel chemin augmentant), le nombre
d'itérations peut dépendre de la valeur du flot maximum, donc des capacités —
ce qui n'est pas une borne polynomiale (on a vu en cours l'exemple en losange
où DFS fait O(F) itérations alors qu'il aurait fallu 2). Edmonds & Karp ont montré
que si on choisit à chaque fois le **plus court** chemin augmentant en nombre
d'arcs (ce que donne BFS), alors le nombre d'augmentations est borné par
`O(V·E)` indépendamment des capacités.

### 2.3 Complexité

- BFS : `O(V + E)`.
- Nombre d'augmentations : `O(V·E)`.
- **Total : `O(V · E²)`.**

L'idée de la preuve (qu'on a vue en cours, on la résume ici) : la distance
`d(s, v)` dans le résiduel ne diminue **jamais** au fil des augmentations. Et
chaque arc devient critique (= bottleneck) au plus `O(V)` fois entre deux
moments où sa direction est inversée. Comme il y a `E` arcs, on a au plus
`O(V·E)` augmentations.

### 2.4 Calcul de la coupe min

Une fois Ford-Fulkerson terminé, on relance un BFS depuis la source sur le
graphe résiduel pour identifier l'ensemble `S` des noeuds atteignables. Tous
les arcs `(u, v)` *réels* avec `u ∈ S` et `v ∉ S` sont nécessairement **saturés**
(sinon le BFS aurait franchi cet arc et `v` serait dans `S`). Ces arcs forment
la coupe minimum, et par le théorème max-flow min-cut, la somme de leurs
capacités est exactement la valeur du flot maximum.

Cette dernière partie m'a pris un peu de temps à bien comprendre la première
fois — surtout pourquoi les arcs *inverses* qui traversent la coupe (de `T`
vers `S`) ne comptent pas dans la coupe. C'est parce qu'on définit la
capacité d'une coupe comme la somme des capacités des arcs allant de `S` vers
`T`, point. Les arcs allant de `T` vers `S` n'ont pas à être saturés.

---

## 3. MCF avec Bellman-Ford

### 3.1 Idée générale (Successive Shortest Paths)

On augmente itérativement le long du **plus court chemin en COÛT** dans le
graphe résiduel, jusqu'à atteindre le volume de flot souhaité ou jusqu'à ce
qu'il n'y ait plus de chemin du source au puits. C'est analogue à Ford-Fulkerson
sauf qu'on choisit le chemin par coût et non par longueur.

### 3.2 Pourquoi Bellman-Ford ?

Le graphe résiduel contient des arcs *inverses* dont le coût est l'opposé du
coût de l'arc réel correspondant. Donc même si le graphe original n'a que des
coûts positifs, dès qu'on commence à pousser du flot, le résiduel contient
des arcs à coût négatif. Dijkstra ne marche pas sur de tels graphes (sauf
si on reweighte — cf. section 4), donc on utilise **Bellman-Ford**, qui gère
naturellement les coûts négatifs.

### 3.3 Pseudo-code

```
fonction MCF-BF(G, s, t, max_flow_optionnel) :
    si G a un cycle négatif :
        avorter "problème mal posé"
    cost, flow = 0, 0
    tant que pas atteint max_flow :
        (dist, parent) = Bellman-Ford(G_résiduel, s)
        si dist[t] = +∞ :
            sortir
        chemin = reconstruire(parent, s, t)
        δ = min des capacités résiduelles sur le chemin
        cost += dist[t] * δ
        flow += δ
        augmenter(chemin, δ)
    renvoyer (cost, flow)
```

### 3.4 Pourquoi un cycle négatif rend le problème mal posé

Si le graphe original contient un cycle de coût strictement négatif, on peut
faire tourner indéfiniment du flot autour de ce cycle (à condition qu'il ait
des capacités) en diminuant le coût total à chaque tour. Le problème de MCF
devient non borné inférieurement, donc mathématiquement il n'y a pas de
solution optimale. On détecte ça en amont avec un Bellman-Ford et on
abandonne proprement avec un message si c'est le cas.

### 3.5 Complexité

- Un Bellman-Ford : `O(V · E)`.
- Nombre d'augmentations : au pire `O(F)` où `F` est la valeur totale du flot
  qu'on pousse (en fait l'argument est plus fin avec les capacités entières mais
  borne raisonnable).
- **Total : `O(F · V · E)`.**

Pour des capacités élevées ou pour pousser de grands volumes, c'est lourd, d'où
la variante Dijkstra+Johnson.

---

## 4. MCF avec Dijkstra + reweighting de Johnson

### 4.1 Idée

On veut utiliser Dijkstra (rapide grâce à `heapq`) au lieu de Bellman-Ford à
chaque itération. Le souci, c'est que Dijkstra ne supporte pas les coûts
négatifs. Solution : reweighter les coûts pour qu'ils soient tous `>= 0`, tout
en préservant les plus courts chemins.

### 4.2 La transformation

On choisit un vecteur de **potentiels** `h : V → R` et on définit les coûts
reweightés :

```
w'(u, v) = w(u, v) + h[u] − h[v]
```

Le long d'un chemin `s → v₁ → v₂ → ... → t`, la somme des `h[u] − h[v]` se
**télescope** :

```
Σ w'(vᵢ, vᵢ₊₁) = ( Σ w(vᵢ, vᵢ₊₁) ) + h[s] − h[t]
```

Autrement dit le coût reweighté d'un chemin est juste le coût réel plus une
constante (qui ne dépend que de `s` et `t`). Donc les plus courts chemins
sont préservés.

### 4.3 Preuve que `w' ≥ 0` après bonne initialisation

**Lemme.** Soit `h` le vecteur des distances depuis `s` dans le graphe résiduel
courant (calculé par Bellman-Ford). Alors pour tout arc `(u, v)` avec capacité
résiduelle `> 0`, on a `w'(u, v) ≥ 0`.

**Preuve.** Par définition d'un plus court chemin, on a `h[v] ≤ h[u] + w(u, v)`
(inégalité triangulaire des distances). Donc `w(u, v) + h[u] − h[v] ≥ 0`,
c'est-à-dire `w'(u, v) ≥ 0`. □

### 4.4 Preuve que `w' ≥ 0` se maintient après une augmentation

Après chaque itération de Dijkstra, on met à jour les potentiels :

```
h'[v] = h[v] + d[v]
```

où `d[v]` est la distance de Dijkstra (sur les coûts reweightés) depuis `s`.
On veut montrer que pour le NOUVEAU résiduel (qui contient des arcs inverses
en plus si des arcs ont été utilisés), `w'_new(u, v) ≥ 0` pour tout arc.

**Pour un arc `(u, v)` déjà présent avant** (capacité résiduelle > 0 dans les
deux résiduels) :

```
w'_new(u, v) = w(u, v) + h'[u] − h'[v]
             = w(u, v) + h[u] + d[u] − h[v] − d[v]
             = w'(u, v) + d[u] − d[v]
```

Et par Dijkstra appliqué à `w'`, on a `d[v] ≤ d[u] + w'(u, v)`, ce qui donne
`w'(u, v) + d[u] − d[v] ≥ 0`. ✓

**Pour un arc *inverse* `(v, u)` qui apparaît à cause de l'augmentation** :
ce nouvel arc inverse existe parce qu'on a utilisé `(u, v)` sur le chemin
augmentant. Comme ce chemin était optimal (Dijkstra), on a
**égalité** dans la relaxation : `d[v] = d[u] + w'(u, v)`. Donc :

```
w'_new(v, u) = w(v, u) + h'[v] − h'[u]
             = −w(u, v) + h[v] + d[v] − h[u] − d[u]
             = −( w(u, v) + h[u] − h[v] ) + (d[v] − d[u])
             = −w'(u, v) + w'(u, v)
             = 0
```

Donc l'arc inverse a un coût reweighté de `0`, ce qui est bien `≥ 0`. ✓ □

Cette propriété nous permet d'enchaîner les Dijkstra sans jamais refaire de
Bellman-Ford après l'initialisation.

### 4.5 Récupération du coût réel

Comme Dijkstra travaille sur les coûts reweightés, `dist[t]` n'est PAS le coût
réel du chemin. On le retrouve par la formule de télescopage :

```
cost_réel(s → t) = dist_reweighted[t] + h[t] − h[s]
```

(Note : `h[s]` reste constant à 0 puisque `d[s] = 0` à chaque Dijkstra, donc
`h[s] += 0`.)

### 4.6 Complexité

- Initialisation par Bellman-Ford : `O(V · E)` (une seule fois).
- Chaque Dijkstra avec un tas binaire `heapq` : `O((V + E) log V)`.
- Nombre d'augmentations : `O(F)` comme avant.
- **Total : `O(V · E + F · (V + E) log V)`.**

Beaucoup mieux que `O(F · V · E)` dès que `log V < V` (donc à peu près tout
le temps).

---

## 5. Détection de cycles négatifs

### 5.1 Méthode principale : Bellman-Ford

Idée standard : après `V−1` itérations de relaxation, on a les plus courts
chemins **si et seulement si** il n'y a pas de cycle négatif accessible depuis
la source. On fait alors une `V`-ième itération : si **une seule** distance
peut encore être améliorée, c'est qu'il y a un cycle négatif.

Pour détecter même les cycles non accessibles depuis une source donnée, on
utilise l'astuce de la **source virtuelle** : on initialise `dist[v] = 0`
pour tous les noeuds, ce qui revient à ajouter implicitement un noeud `s*`
relié à tous avec coût 0.

### 5.2 Extraction du cycle

Une fois détecté qu'un noeud `v` se fait encore relaxer à la `V`-ième
itération, on remonte `V` fois via `parent[v]`. Cette astuce (vue en cours)
garantit qu'on tombe **dans** le cycle (et pas sur la "queue" qui mène au
cycle), parce qu'en `V` pas on a forcément fait au moins un tour complet du
cycle. Ensuite on suit `parent` jusqu'à retomber sur le noeud de départ pour
extraire la séquence des arcs.

### 5.3 Méthode de fallback : DFS

On a aussi implémenté une variante DFS qui trouve UN cycle quelconque dans
le graphe (via la détection classique d'arête arrière sur un sommet gris)
puis vérifie son poids. C'est moins puissant que Bellman-Ford : on ne trouve
que le premier cycle rencontré, et si celui-ci n'est pas négatif on peut
manquer un cycle négatif ailleurs. En pratique on l'utilise comme double-check
sur de petits graphes.

Pour info, il existe des algorithmes plus sophistiqués (genre SPFA avec
détection, ou Howard's algorithm pour le cycle de moyenne minimale), mais
Bellman-Ford suffit largement pour notre projet.

---

## 6. Comparaison des deux approches MCF

### 6.1 Tableau récapitulatif

| Critère                        | MCF Bellman-Ford       | MCF Dijkstra + Johnson           |
|--------------------------------|------------------------|----------------------------------|
| Complexité par itération       | `O(V · E)`             | `O((V + E) log V)`               |
| Complexité totale              | `O(F · V · E)`         | `O(V·E + F · (V+E) log V)`       |
| Gère les coûts négatifs ?      | Oui, directement       | Oui, après reweighting initial   |
| Implémentation                 | Plus simple            | Un peu plus subtile (potentiels) |
| Robustesse aux cas tordus      | Très robuste           | Sensible aux noeuds inaccessibles|

### 6.2 Discussion

Honnêtement, sur les petits graphes des tests, on ne voit pas la différence
de performance : les deux algos finissent en quelques millisecondes. La vraie
différence vient avec des capacités élevées (donc beaucoup d'augmentations)
ou avec des graphes denses : là, le facteur `log V` au lieu de `V·E` par
itération devient très visible.

Le code de la version Dijkstra+Johnson est un peu plus délicat parce qu'il
faut gérer correctement :

- La **mise à jour des potentiels** après chaque Dijkstra (sinon on perd
  l'invariant `w' ≥ 0`).
- Le cas des **noeuds inaccessibles** (potentiel `+∞`). On a choisi de les
  skipper dans Dijkstra plutôt que de les "patcher" à un grand entier — c'est
  cohérent tant que les noeuds restent inaccessibles ; si certains deviennent
  accessibles ultérieurement via des arcs inverses, ça poserait problème. Pour
  les tests qu'on propose, tous les noeuds sont toujours atteignables depuis la
  source dans le résiduel, donc on est tranquille.
- La **récupération du coût réel** à partir de `dist[sink] + h[sink] − h[source]`
  — pas évident la première fois qu'on lit la formule, mais c'est juste le
  télescopage.

Au final on a fait tourner les deux algos sur les mêmes données dans les tests
et on vérifie qu'ils donnent **exactement** le même résultat (coût et flot total),
ce qui est la meilleure sanity check qu'on a.

### 6.3 Quand préférer l'une ou l'autre ?

- **Bellman-Ford** : si on veut un code court et qu'on n'a pas de soucis de
  performance, ou si on veut une implémentation très robuste qui marche
  partout sans se poser de questions. C'est aussi plus simple à expliquer en
  soutenance.
- **Dijkstra + Johnson** : si on travaille sur des instances plus grosses ou
  si on cherche à se rapprocher des perfs des solveurs industriels. Toutes les
  vraies implémentations de MCF (LEMON, OR-Tools, etc.) utilisent une variante
  de cette idée.

---

## Conclusion

Le projet nous a permis de revisiter en pratique des algos qu'on avait vus en
cours uniquement de manière théorique. La partie sur le graphe résiduel — en
particulier le truc du flot symétrique pour avoir une formule unique de
capacité résiduelle — est probablement ce qu'on garde le plus comme leçon
d'ingénierie : c'est exactement le genre de petit choix qui simplifie tout
le reste du code.

La partie sur Johnson nous a aussi obligés à bien comprendre POURQUOI les
potentiels marchent, plutôt que de juste appliquer la formule. La preuve que
`w' ≥ 0` se maintient à travers les itérations est subtile mais
satisfaisante une fois qu'on l'a en main.

Si on devait pousser plus loin, on essaierait :

- D'implémenter la variante par annulation de cycles négatifs (Klein) pour
  comparer avec les augmentations successives.
- D'ajouter un benchmark sur des graphes plus gros (générés aléatoirement) pour
  voir empiriquement la différence BF vs Dijkstra.
- De gérer proprement les arcs antiparallèles, qu'on a juste interdits ici par
  facilité.

---
