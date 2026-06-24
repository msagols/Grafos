"""
Mini proyecto de grafos + visualizaciones + animaciones

Incluye:
1) Generar grafos r-regulares de orden n
2) Generar un grafo dada una sucesión gráfica
3) Kruskal
4) Kruskal inverso / reverse-delete
5) Prim

Además:
- genera PNGs bonitos
- genera GIFs animados para capturas / README de GitHub
- guarda todo en outputs/ junto a este archivo .py

Dependencias:
    pip install networkx matplotlib pillow
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# Configuración general
# ============================================================

GENERAR_GIFS = True
GENERAR_PNGS = True

GIF_FPS = 1.2
GIF_DPI = 130

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"


# ============================================================
# Tipos
# ============================================================

Edge = Tuple[int, int]
WeightedEdge = Tuple[int, int, int]
Adj = Dict[int, Set[int]]
WeightedAdj = Dict[int, List[Tuple[int, int]]]


# ============================================================
# Utilidades base
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_edge(u: int, v: int) -> Edge:
    """
    Guarda una arista siempre como (menor, mayor).

    Así evitamos que (2, 5) y (5, 2) se traten como cosas diferentes.
    """
    return (u, v) if u < v else (v, u)


def empty_graph(n: int) -> Adj:
    """Crea un grafo vacío con vértices 0, 1, ..., n-1."""
    return {i: set() for i in range(n)}


def add_edge(g: Adj, u: int, v: int) -> None:
    """Agrega arista no dirigida."""
    if u == v:
        return
    g[u].add(v)
    g[v].add(u)


def edge_list(g: Adj) -> List[Edge]:
    """Regresa las aristas del grafo sin repetir."""
    edges = set()
    for u in g:
        for v in g[u]:
            edges.add(normalize_edge(u, v))
    return sorted(edges)


def degrees(g: Adj) -> List[int]:
    """Regresa los grados de los vértices."""
    return [len(g[v]) for v in sorted(g)]


def weighted_adj_from_edges(n: int, edges: List[WeightedEdge]) -> WeightedAdj:
    """Convierte lista de aristas pesadas a lista de adyacencia."""
    adj: WeightedAdj = {i: [] for i in range(n)}

    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    return adj


def weighted_edge_key(edge: WeightedEdge) -> Edge:
    """Llave sin peso para comparar aristas pesadas."""
    u, v, _ = edge
    return normalize_edge(u, v)


def total_weight(edges: List[WeightedEdge]) -> int:
    return sum(w for _, _, w in edges)


# ============================================================
# 1) Generar grafo r-regular
# ============================================================

def generate_r_regular_graph(n: int, r: int, max_tries: int = 10000) -> Adj:
    """
    Genera un grafo simple r-regular de orden n.

    Un grafo r-regular significa:
        todos los vértices tienen grado r.

    Método usado:
    - hago r copias de cada vértice
    - revuelvo todas esas copias
    - emparejo de dos en dos
    - si sale lazo o arista repetida, reinicio

    Es un método random sencillo. Para ejemplos pequeños funciona bien.
    """
    if r < 0:
        raise ValueError("r no puede ser negativo.")

    if r >= n:
        raise ValueError("Para grafo simple debe cumplirse r < n.")

    if (n * r) % 2 != 0:
        raise ValueError("No existe tal grafo: n*r debe ser par.")

    for _ in range(max_tries):
        stubs = []

        for v in range(n):
            stubs.extend([v] * r)

        random.shuffle(stubs)

        g = empty_graph(n)
        ok = True

        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]

            if u == v or v in g[u]:
                ok = False
                break

            add_edge(g, u, v)

        if ok and all(len(g[v]) == r for v in g):
            return g

    raise RuntimeError(
        "No se pudo generar el grafo r-regular. "
        "Prueba con otros valores o sube max_tries."
    )


# ============================================================
# 2) Grafo desde sucesión gráfica
# ============================================================

def graph_from_degree_sequence(sequence: List[int]) -> Adj:
    """
    Construye un grafo simple dada una sucesión gráfica.

    Se usa Havel-Hakimi.

    Idea:
    - ordeno los grados pendientes
    - tomo el vértice que más conexiones necesita
    - lo conecto con los siguientes de mayor grado pendiente
    - resto grados
    - repito
    """
    n = len(sequence)

    if any(d < 0 for d in sequence):
        raise ValueError("Los grados no pueden ser negativos.")

    if any(d >= n for d in sequence):
        raise ValueError("En grafo simple, ningún grado puede ser >= n.")

    if sum(sequence) % 2 != 0:
        raise ValueError("La suma de grados debe ser par.")

    g = empty_graph(n)

    # Cada elemento es [grado_pendiente, vértice]
    remaining = [[sequence[i], i] for i in range(n)]

    while True:
        remaining.sort(reverse=True)

        if remaining[0][0] == 0:
            return g

        d, v = remaining[0]
        remaining = remaining[1:]

        if d > len(remaining):
            raise ValueError("La sucesión no es gráfica.")

        for i in range(d):
            remaining[i][0] -= 1
            u = remaining[i][1]

            if remaining[i][0] < 0:
                raise ValueError("La sucesión no es gráfica.")

            add_edge(g, v, u)


# ============================================================
# Union-Find para Kruskal
# ============================================================

@dataclass
class DSU:
    parent: List[int]
    rank: List[int]

    @classmethod
    def create(cls, n: int) -> "DSU":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        """
        Encuentra el representante del conjunto.

        También hace compresión de caminos, que básicamente aplana
        la estructura para que las siguientes consultas sean rápidas.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """
        Une los conjuntos de a y b.

        Regresa:
        - True si sí se unieron
        - False si ya estaban en el mismo conjunto
        """
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        return True


# ============================================================
# 3) Kruskal
# ============================================================

def kruskal(n: int, edges: List[WeightedEdge]) -> Tuple[List[WeightedEdge], int]:
    """
    Algoritmo de Kruskal.

    Idea:
    - ordeno aristas de menor a mayor peso
    - voy metiendo las más baratas
    - pero solo si no forman ciclo
    """
    dsu = DSU.create(n)
    mst: List[WeightedEdge] = []
    total = 0

    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(u, v):
            mst.append((u, v, w))
            total += w

        if len(mst) == n - 1:
            break

    if len(mst) != n - 1:
        raise ValueError("El grafo no es conexo.")

    return mst, total


def kruskal_steps(n: int, edges: List[WeightedEdge]) -> List[dict]:
    """
    Genera pasos para animar Kruskal.

    Cada paso guarda:
    - arista actual
    - aristas aceptadas
    - aristas rechazadas
    - texto explicativo
    """
    dsu = DSU.create(n)

    accepted: List[WeightedEdge] = []
    rejected: List[WeightedEdge] = []

    steps = [
        {
            "current": None,
            "accepted": accepted[:],
            "rejected": rejected[:],
            "text": "Inicio: ordenamos aristas por peso."
        }
    ]

    for edge in sorted(edges, key=lambda e: e[2]):
        u, v, w = edge

        if dsu.union(u, v):
            accepted.append(edge)
            text = f"Se acepta {u}-{v} con peso {w}."
        else:
            rejected.append(edge)
            text = f"Se rechaza {u}-{v} porque forma ciclo."

        steps.append(
            {
                "current": edge,
                "accepted": accepted[:],
                "rejected": rejected[:],
                "text": text
            }
        )

        if len(accepted) == n - 1:
            break

    steps.append(
        {
            "current": None,
            "accepted": accepted[:],
            "rejected": rejected[:],
            "text": f"Final: MST encontrado con peso total {total_weight(accepted)}."
        }
    )

    return steps


# ============================================================
# 4) Kruskal inverso / reverse-delete
# ============================================================

def is_connected(n: int, edges: List[WeightedEdge]) -> bool:
    """Revisa si un grafo pesado es conexo. El peso no importa aquí."""
    if n <= 1:
        return True

    adj = weighted_adj_from_edges(n, edges)

    visited = set()
    stack = [0]

    while stack:
        u = stack.pop()

        if u in visited:
            continue

        visited.add(u)

        for v, _ in adj[u]:
            if v not in visited:
                stack.append(v)

    return len(visited) == n


def kruskal_inverso(n: int, edges: List[WeightedEdge]) -> Tuple[List[WeightedEdge], int]:
    """
    Kruskal inverso, también llamado reverse-delete.

    Idea:
    - ordeno aristas de mayor a menor peso
    - intento borrar las caras
    - si el grafo sigue conexo, la arista se va
    - si el grafo se desconecta, la arista se queda
    """
    if not is_connected(n, edges):
        raise ValueError("El grafo no es conexo.")

    remaining = edges[:]

    for edge in sorted(edges, key=lambda e: e[2], reverse=True):
        candidate = remaining[:]
        candidate.remove(edge)

        if is_connected(n, candidate):
            remaining = candidate

        if len(remaining) == n - 1:
            break

    return sorted(remaining), total_weight(remaining)


def kruskal_inverso_steps(n: int, edges: List[WeightedEdge]) -> List[dict]:
    """
    Genera pasos para animar Kruskal inverso.
    """
    if not is_connected(n, edges):
        raise ValueError("El grafo no es conexo.")

    remaining = edges[:]
    deleted: List[WeightedEdge] = []
    kept: List[WeightedEdge] = []

    steps = [
        {
            "current": None,
            "remaining": remaining[:],
            "deleted": deleted[:],
            "kept": kept[:],
            "text": "Inicio: ordenamos aristas de mayor a menor peso."
        }
    ]

    for edge in sorted(edges, key=lambda e: e[2], reverse=True):
        u, v, w = edge

        candidate = remaining[:]
        candidate.remove(edge)

        if is_connected(n, candidate):
            remaining = candidate
            deleted.append(edge)
            text = f"Se elimina {u}-{v} con peso {w}; el grafo sigue conexo."
        else:
            kept.append(edge)
            text = f"Se conserva {u}-{v} con peso {w}; si se quita, desconecta."

        steps.append(
            {
                "current": edge,
                "remaining": remaining[:],
                "deleted": deleted[:],
                "kept": kept[:],
                "text": text
            }
        )

        if len(remaining) == n - 1:
            break

    steps.append(
        {
            "current": None,
            "remaining": remaining[:],
            "deleted": deleted[:],
            "kept": kept[:],
            "text": f"Final: MST encontrado con peso total {total_weight(remaining)}."
        }
    )

    return steps


# ============================================================
# 5) Prim
# ============================================================

def prim(n: int, edges: List[WeightedEdge], start: int = 0) -> Tuple[List[WeightedEdge], int]:
    """
    Algoritmo de Prim.

    Idea:
    - empiezo en un vértice
    - en cada paso busco la arista más barata que salga del conjunto visitado
    - agrego esa arista y el nuevo vértice
    """
    adj = weighted_adj_from_edges(n, edges)

    visited = {start}
    mst: List[WeightedEdge] = []
    total = 0

    while len(visited) < n:
        best: Optional[WeightedEdge] = None

        for u in visited:
            for v, w in adj[u]:
                if v not in visited:
                    candidate = (u, v, w)

                    if best is None or w < best[2]:
                        best = candidate

        if best is None:
            raise ValueError("El grafo no es conexo.")

        u, v, w = best
        visited.add(v)
        mst.append(best)
        total += w

    return mst, total


def prim_steps(n: int, edges: List[WeightedEdge], start: int = 0) -> List[dict]:
    """
    Genera pasos para animar Prim.
    """
    adj = weighted_adj_from_edges(n, edges)

    visited = {start}
    mst: List[WeightedEdge] = []

    steps = [
        {
            "visited": visited.copy(),
            "mst": mst[:],
            "chosen": None,
            "candidates": [],
            "text": f"Inicio: empezamos en el vértice {start}."
        }
    ]

    while len(visited) < n:
        candidates: List[WeightedEdge] = []

        for u in visited:
            for v, w in adj[u]:
                if v not in visited:
                    candidates.append((u, v, w))

        if not candidates:
            raise ValueError("El grafo no es conexo.")

        chosen = min(candidates, key=lambda e: e[2])

        u, v, w = chosen
        visited.add(v)
        mst.append(chosen)

        steps.append(
            {
                "visited": visited.copy(),
                "mst": mst[:],
                "chosen": chosen,
                "candidates": candidates[:],
                "text": f"Se agrega {u}-{v} con peso {w}."
            }
        )

    steps.append(
        {
            "visited": visited.copy(),
            "mst": mst[:],
            "chosen": None,
            "candidates": [],
            "text": f"Final: MST encontrado con peso total {total_weight(mst)}."
        }
    )

    return steps


# ============================================================
# Generador de grafo pesado random
# ============================================================

def random_connected_weighted_graph(
    n: int,
    extra_edges: int,
    min_w: int = 1,
    max_w: int = 20
) -> List[WeightedEdge]:
    """
    Genera un grafo conexo pesado.

    Primero hago un árbol random para asegurar conexión.
    Luego meto aristas extra al azar.
    """
    edges_set = set()

    # Árbol random base.
    for v in range(1, n):
        u = random.randint(0, v - 1)
        edges_set.add(normalize_edge(u, v))

    # Aristas extra posibles.
    possible = []

    for u in range(n):
        for v in range(u + 1, n):
            e = normalize_edge(u, v)

            if e not in edges_set:
                possible.append(e)

    random.shuffle(possible)

    for e in possible[:extra_edges]:
        edges_set.add(e)

    weighted_edges = []

    for u, v in sorted(edges_set):
        w = random.randint(min_w, max_w)
        weighted_edges.append((u, v, w))

    return weighted_edges


# ============================================================
# Conversiones a NetworkX
# ============================================================

def adj_to_nx(g: Adj) -> nx.Graph:
    G = nx.Graph()

    for u in g:
        G.add_node(u)

    for u in g:
        for v in g[u]:
            if u < v:
                G.add_edge(u, v)

    return G


def weighted_edges_to_nx(n: int, edges: List[WeightedEdge]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G


# ============================================================
# Visualizaciones estáticas
# ============================================================

def draw_unweighted_graph(
    g: Adj,
    title: str,
    output_path: Path,
    seed_layout: int = 42,
    node_color: str = "#8ecae6"
) -> None:
    """
    Dibuja un grafo no pesado en PNG.
    """
    G = adj_to_nx(g)
    pos = nx.spring_layout(G, seed=seed_layout)

    plt.figure(figsize=(9, 7))
    plt.title(title, fontsize=16, fontweight="bold")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1000,
        node_color=node_color,
        edgecolors="#1d3557",
        linewidths=2
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=2.5,
        edge_color="#457b9d"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_weight="bold",
        font_color="#111111"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def draw_weighted_graph(
    n: int,
    edges: List[WeightedEdge],
    title: str,
    output_path: Path,
    highlight_edges: Optional[List[WeightedEdge]] = None,
    seed_layout: int = 42
) -> None:
    """
    Dibuja grafo pesado.

    Si highlight_edges trae algo, esas aristas se pintan en verde.
    """
    G = weighted_edges_to_nx(n, edges)
    pos = nx.spring_layout(G, seed=seed_layout)

    highlight_set = set()

    if highlight_edges is not None:
        highlight_set = {weighted_edge_key(e) for e in highlight_edges}

    normal_edges = []
    special_edges = []

    for u, v in G.edges():
        key = normalize_edge(u, v)

        if key in highlight_set:
            special_edges.append((u, v))
        else:
            normal_edges.append((u, v))

    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=16, fontweight="bold")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1000,
        node_color="#f1fa8c",
        edgecolors="#444444",
        linewidths=2
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        width=2.2,
        edge_color="#bdbdbd"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=special_edges,
        width=4,
        edge_color="#2a9d8f"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_weight="bold",
        font_color="#111111"
    )

    labels = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=labels,
        font_size=11,
        font_color="#d62828",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# Helpers para animaciones
# ============================================================

def draw_weighted_state(
    ax,
    n: int,
    edges: List[WeightedEdge],
    pos,
    title: str,
    subtitle: str = "",
    green_edges: Optional[List[WeightedEdge]] = None,
    orange_edges: Optional[List[WeightedEdge]] = None,
    red_edges: Optional[List[WeightedEdge]] = None,
    blue_nodes: Optional[Set[int]] = None,
    faded_edges: Optional[List[WeightedEdge]] = None
) -> None:
    """
    Dibuja un estado del grafo pesado para animaciones.

    Colores:
    - verde: aristas elegidas / MST parcial
    - naranja: arista actual o candidatas importantes
    - rojo: aristas rechazadas o eliminadas
    - azul: nodos visitados en Prim
    """
    ax.clear()

    G = weighted_edges_to_nx(n, edges)

    green_set = {weighted_edge_key(e) for e in green_edges or []}
    orange_set = {weighted_edge_key(e) for e in orange_edges or []}
    red_set = {weighted_edge_key(e) for e in red_edges or []}
    faded_set = {weighted_edge_key(e) for e in faded_edges or []}

    normal = []
    green = []
    orange = []
    red = []
    faded = []

    for u, v in G.edges():
        key = normalize_edge(u, v)

        if key in green_set:
            green.append((u, v))
        elif key in orange_set:
            orange.append((u, v))
        elif key in red_set:
            red.append((u, v))
        elif key in faded_set:
            faded.append((u, v))
        else:
            normal.append((u, v))

    node_colors = []

    for node in G.nodes():
        if blue_nodes is not None and node in blue_nodes:
            node_colors.append("#90e0ef")
        else:
            node_colors.append("#f1fa8c")

    ax.set_title(title, fontsize=15, fontweight="bold")

    if subtitle:
        ax.text(
            0.5,
            -0.06,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.45")
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=950,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=2
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=normal,
        width=2,
        edge_color="#c7c7c7"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=faded,
        width=1.5,
        edge_color="#dddddd",
        style="dashed",
        alpha=0.35
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=orange,
        width=4,
        edge_color="#f4a261"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=red,
        width=3,
        edge_color="#e63946",
        style="dashed"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=green,
        width=4.5,
        edge_color="#2a9d8f"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=12,
        font_weight="bold",
        font_color="#111111"
    )

    labels = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        edge_labels=labels,
        font_size=10,
        font_color="#d62828",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
    )

    ax.axis("off")


# ============================================================
# Animaciones de grafos no pesados
# ============================================================

def animate_unweighted_reveal(
    g: Adj,
    title: str,
    output_path: Path,
    seed_layout: int = 42,
    node_color: str = "#8ecae6"
) -> None:
    """
    GIF donde las aristas van apareciendo una por una.
    """
    G = adj_to_nx(g)
    pos = nx.spring_layout(G, seed=seed_layout)
    edges = list(G.edges())

    fig, ax = plt.subplots(figsize=(9, 7))

    def update(frame: int):
        ax.clear()

        shown_edges = edges[:frame]

        ax.set_title(title, fontsize=15, fontweight="bold")

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=1000,
            node_color=node_color,
            edgecolors="#1d3557",
            linewidths=2
        )

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=shown_edges,
            width=2.8,
            edge_color="#457b9d"
        )

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=12,
            font_weight="bold"
        )

        ax.text(
            0.5,
            -0.06,
            f"Aristas mostradas: {len(shown_edges)} de {len(edges)}",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.45")
        )

        ax.axis("off")

    frames = len(edges) + 1
    anim = FuncAnimation(fig, update, frames=frames, interval=800, repeat_delay=1200)

    anim.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(fig)


# ============================================================
# Animación de Kruskal
# ============================================================

def animate_kruskal(
    n: int,
    edges: List[WeightedEdge],
    output_path: Path,
    seed_layout: int = 42
) -> None:
    G = weighted_edges_to_nx(n, edges)
    pos = nx.spring_layout(G, seed=seed_layout)

    steps = kruskal_steps(n, edges)

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame: int):
        step = steps[frame]

        current = step["current"]
        orange = [current] if current is not None else []

        draw_weighted_state(
            ax=ax,
            n=n,
            edges=edges,
            pos=pos,
            title="Kruskal",
            subtitle=step["text"],
            green_edges=step["accepted"],
            orange_edges=orange,
            red_edges=step["rejected"]
        )

    anim = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat_delay=1500)

    anim.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(fig)


# ============================================================
# Animación de Kruskal inverso
# ============================================================

def animate_kruskal_inverso(
    n: int,
    edges: List[WeightedEdge],
    output_path: Path,
    seed_layout: int = 42
) -> None:
    G = weighted_edges_to_nx(n, edges)
    pos = nx.spring_layout(G, seed=seed_layout)

    steps = kruskal_inverso_steps(n, edges)

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame: int):
        step = steps[frame]

        current = step["current"]
        orange = [current] if current is not None else []

        # En Kruskal inverso:
        # - remaining son las que siguen vivas
        # - deleted son las que ya se borraron
        # - current va naranja
        # - al final remaining queda verde
        if frame == len(steps) - 1:
            green_edges = step["remaining"]
        else:
            green_edges = []

        draw_weighted_state(
            ax=ax,
            n=n,
            edges=edges,
            pos=pos,
            title="Kruskal inverso / reverse-delete",
            subtitle=step["text"],
            green_edges=green_edges,
            orange_edges=orange,
            red_edges=step["deleted"],
            faded_edges=step["deleted"]
        )

    anim = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat_delay=1500)

    anim.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(fig)


# ============================================================
# Animación de Prim
# ============================================================

def animate_prim(
    n: int,
    edges: List[WeightedEdge],
    output_path: Path,
    seed_layout: int = 42,
    start: int = 0
) -> None:
    G = weighted_edges_to_nx(n, edges)
    pos = nx.spring_layout(G, seed=seed_layout)

    steps = prim_steps(n, edges, start=start)

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame: int):
        step = steps[frame]

        chosen = step["chosen"]
        orange = [chosen] if chosen is not None else []

        draw_weighted_state(
            ax=ax,
            n=n,
            edges=edges,
            pos=pos,
            title="Prim",
            subtitle=step["text"],
            green_edges=step["mst"],
            orange_edges=orange,
            blue_nodes=step["visited"]
        )

    anim = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat_delay=1500)

    anim.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(fig)


# ============================================================
# Reportes en texto
# ============================================================

def format_unweighted_graph_report(title: str, g: Adj) -> str:
    lines = []
    lines.append(title)
    lines.append("-" * len(title))
    lines.append(f"Vértices: {list(g.keys())}")
    lines.append(f"Aristas : {edge_list(g)}")
    lines.append(f"Grados  : {degrees(g)}")
    lines.append("")
    return "\n".join(lines)


def format_weighted_graph_report(title: str, n: int, edges: List[WeightedEdge]) -> str:
    lines = []
    lines.append(title)
    lines.append("-" * len(title))
    lines.append(f"Vértices: {list(range(n))}")
    lines.append("Aristas con peso:")

    for u, v, w in sorted(edges):
        lines.append(f"  {u} -- {v}   peso={w}")

    lines.append("")
    return "\n".join(lines)


def format_mst_report(title: str, mst: List[WeightedEdge], total: int) -> str:
    lines = []
    lines.append(title)
    lines.append("-" * len(title))

    for u, v, w in sorted(mst):
        lines.append(f"  {u} -- {v}   peso={w}")

    lines.append(f"Peso total = {total}")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Corrida completa
# ============================================================

def run_demo(seed: int, base_output_dir: Path = OUTPUTS_DIR) -> None:
    random.seed(seed)

    run_dir = base_output_dir / f"run_seed_{seed}"
    ensure_dir(run_dir)

    report_parts = []
    report_parts.append("=" * 72)
    report_parts.append(f"CORRIDA RANDOM - seed = {seed}")
    report_parts.append("=" * 72)
    report_parts.append("")

    # --------------------------------------------------------
    # 1) Grafo r-regular
    # --------------------------------------------------------
    n_regular = 8
    r_regular = 3

    g_regular = generate_r_regular_graph(n=n_regular, r=r_regular)

    report_parts.append(
        format_unweighted_graph_report(
            f"1) Grafo {r_regular}-regular de orden {n_regular}",
            g_regular
        )
    )

    if GENERAR_PNGS:
        draw_unweighted_graph(
            g_regular,
            title=f"Grafo {r_regular}-regular de orden {n_regular}",
            output_path=run_dir / "01_grafo_regular.png",
            seed_layout=seed,
            node_color="#90e0ef"
        )

    if GENERAR_GIFS:
        animate_unweighted_reveal(
            g_regular,
            title=f"Grafo {r_regular}-regular de orden {n_regular}",
            output_path=run_dir / "01_grafo_regular_animado.gif",
            seed_layout=seed,
            node_color="#90e0ef"
        )

    # --------------------------------------------------------
    # 2) Grafo desde sucesión gráfica
    # --------------------------------------------------------
    sequences = [
        [3, 3, 2, 2, 2, 2],
        [4, 3, 3, 2, 2, 2, 2],
        [3, 3, 3, 3, 2, 2, 2, 2],
    ]

    seq = random.choice(sequences)
    g_seq = graph_from_degree_sequence(seq)

    report_parts.append(
        format_unweighted_graph_report(
            f"2) Grafo generado desde la sucesión gráfica {seq}",
            g_seq
        )
    )

    if GENERAR_PNGS:
        draw_unweighted_graph(
            g_seq,
            title=f"Grafo desde sucesión gráfica {seq}",
            output_path=run_dir / "02_grafo_sucesion.png",
            seed_layout=seed + 1,
            node_color="#ffafcc"
        )

    if GENERAR_GIFS:
        animate_unweighted_reveal(
            g_seq,
            title=f"Grafo desde sucesión gráfica {seq}",
            output_path=run_dir / "02_grafo_sucesion_animado.gif",
            seed_layout=seed + 1,
            node_color="#ffafcc"
        )

    # --------------------------------------------------------
    # 3) Grafo pesado random
    # --------------------------------------------------------
    n_weighted = 7
    extra_edges = 6

    weighted_edges = random_connected_weighted_graph(
        n=n_weighted,
        extra_edges=extra_edges,
        min_w=1,
        max_w=25
    )

    report_parts.append(
        format_weighted_graph_report(
            "3) Grafo pesado random para MST",
            n_weighted,
            weighted_edges
        )
    )

    layout_seed = seed + 2

    if GENERAR_PNGS:
        draw_weighted_graph(
            n=n_weighted,
            edges=weighted_edges,
            title="Grafo pesado random",
            output_path=run_dir / "03_grafo_pesado.png",
            seed_layout=layout_seed
        )

    # --------------------------------------------------------
    # 4) Kruskal
    # --------------------------------------------------------
    mst_k, total_k = kruskal(n_weighted, weighted_edges)

    report_parts.append(
        format_mst_report(
            "4) Kruskal",
            mst_k,
            total_k
        )
    )

    if GENERAR_PNGS:
        draw_weighted_graph(
            n=n_weighted,
            edges=weighted_edges,
            title=f"Kruskal - MST, peso total = {total_k}",
            output_path=run_dir / "04_mst_kruskal.png",
            highlight_edges=mst_k,
            seed_layout=layout_seed
        )

    if GENERAR_GIFS:
        animate_kruskal(
            n=n_weighted,
            edges=weighted_edges,
            output_path=run_dir / "04_kruskal_animado.gif",
            seed_layout=layout_seed
        )

    # --------------------------------------------------------
    # 5) Kruskal inverso
    # --------------------------------------------------------
    mst_ki, total_ki = kruskal_inverso(n_weighted, weighted_edges)

    report_parts.append(
        format_mst_report(
            "5) Kruskal inverso",
            mst_ki,
            total_ki
        )
    )

    if GENERAR_PNGS:
        draw_weighted_graph(
            n=n_weighted,
            edges=weighted_edges,
            title=f"Kruskal inverso - MST, peso total = {total_ki}",
            output_path=run_dir / "05_mst_kruskal_inverso.png",
            highlight_edges=mst_ki,
            seed_layout=layout_seed
        )

    if GENERAR_GIFS:
        animate_kruskal_inverso(
            n=n_weighted,
            edges=weighted_edges,
            output_path=run_dir / "05_kruskal_inverso_animado.gif",
            seed_layout=layout_seed
        )

    # --------------------------------------------------------
    # 6) Prim
    # --------------------------------------------------------
    mst_p, total_p = prim(n_weighted, weighted_edges, start=0)

    report_parts.append(
        format_mst_report(
            "6) Prim",
            mst_p,
            total_p
        )
    )

    if GENERAR_PNGS:
        draw_weighted_graph(
            n=n_weighted,
            edges=weighted_edges,
            title=f"Prim - MST, peso total = {total_p}",
            output_path=run_dir / "06_mst_prim.png",
            highlight_edges=mst_p,
            seed_layout=layout_seed
        )

    if GENERAR_GIFS:
        animate_prim(
            n=n_weighted,
            edges=weighted_edges,
            output_path=run_dir / "06_prim_animado.gif",
            seed_layout=layout_seed,
            start=0
        )

    # --------------------------------------------------------
    # Comparación final
    # --------------------------------------------------------
    report_parts.append("Comparación final")
    report_parts.append("-----------------")
    report_parts.append(f"Kruskal         : {total_k}")
    report_parts.append(f"Kruskal inverso : {total_ki}")
    report_parts.append(f"Prim            : {total_p}")
    report_parts.append("")

    if total_k == total_ki == total_p:
        report_parts.append("Los tres algoritmos dieron el mismo peso total.")
    else:
        report_parts.append("Ojo: los algoritmos dieron pesos distintos.")
        report_parts.append("Eso indicaría un error o un caso que hay que revisar.")

    report_parts.append("")

    report_text = "\n".join(report_parts)

    report_path = run_dir / "reporte_corrida.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"Se guardaron imágenes, GIFs y reporte en:")
    print(run_dir.resolve())


# ============================================================
# Varias corridas
# ============================================================

def run_many_demos(seeds: List[int]) -> None:
    ensure_dir(OUTPUTS_DIR)

    for seed in seeds:
        run_demo(seed)
        print("\n" + "#" * 80 + "\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    """
    Puedes cambiar estas seeds.

    Cada seed crea una carpeta:
        outputs/run_seed_2026/
        outputs/run_seed_7/
        outputs/run_seed_123/

    Dentro de cada carpeta van:
        - PNGs
        - GIFs animados
        - reporte_corrida.txt
    """

    seeds = [2026, 7, 123]

    run_many_demos(seeds)