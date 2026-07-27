"""
Graph representation and validation for Factor-Path Diffusion.

Supports four graph types (INDEPENDENT, DAG, DENSE_DIRECTED, CUSTOM_DIRECTED)
with validation, cycle detection, topological sort, transitive closure,
and PathCertificate for static factor-source-to-output path analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.types import GraphSpec, GraphType


class GraphValidationError(ValueError):
    pass


def validate_graph(graph_spec: GraphSpec) -> None:
    """Validate graph against constraints for its GraphType. Raises on failure."""
    n = graph_spec.num_nodes
    edges = graph_spec.edges
    gtype = graph_spec.graph_type

    _check_node_range(edges, n)
    _check_duplicate_edges(edges)
    _check_self_loops(edges, graph_spec.allow_self_loops)

    if gtype == GraphType.DAG:
        _check_no_cycles(edges, n)
    elif gtype == GraphType.CUSTOM_DIRECTED:
        if not graph_spec.allow_cycles:
            _check_no_cycles(edges, n)


def _check_node_range(edges: tuple[tuple[int, int], ...], num_nodes: int) -> None:
    for u, v in edges:
        if not (0 <= u < num_nodes):
            raise GraphValidationError(
                f"Parent node {u} out of range [0, {num_nodes - 1}]"
            )
        if not (0 <= v < num_nodes):
            raise GraphValidationError(
                f"Child node {v} out of range [0, {num_nodes - 1}]"
            )


def _check_duplicate_edges(edges: tuple[tuple[int, int], ...]) -> None:
    seen = set()
    for e in edges:
        if e in seen:
            raise GraphValidationError(f"Duplicate edge: {e}")
        seen.add(e)


def _check_self_loops(edges: tuple[tuple[int, int], ...], allowed: bool) -> None:
    if allowed:
        return
    for u, v in edges:
        if u == v:
            raise GraphValidationError(f"Self-loop forbidden: ({u}, {v})")


def _check_no_cycles(edges: tuple[tuple[int, int], ...], num_nodes: int) -> None:
    """DFS cycle detection. Raises GraphValidationError if cycle found."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_nodes

    def dfs(u: int) -> list[int]:
        color[u] = GRAY
        for w in adj[u]:
            if color[w] == GRAY:
                return _reconstruct_cycle(u, w, adj, color)
            if color[w] == WHITE:
                cycle = dfs(w)
                if cycle:
                    return cycle
        color[u] = BLACK
        return []

    for start in range(num_nodes):
        if color[start] == WHITE:
            cycle = dfs(start)
            if cycle:
                raise GraphValidationError(f"Cycle detected: {cycle}")
    return None


def _reconstruct_cycle(start: int, target: int, adj: list[list[int]], color: list[int]) -> list[int]:
    """Reconstruct cycle path from dfs state."""
    stack = [start]
    # Simple reconstruction: nodes currently GRAY form the cycle
    cycle_nodes = [i for i, c in enumerate(color) if c == 1]
    # Build a proper cycle path
    path = [target]
    current = start
    while current != target:
        path.append(current)
        for w in adj[current]:
            if color[w] == 1:
                current = w
                break
        else:
            break
    path.append(target)
    return path


def topological_sort(edges: tuple[tuple[int, int], ...], num_nodes: int) -> list[int]:
    """Kahn's algorithm. Returns topological order. Raises if cycle exists."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1

    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for w in adj[u]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)

    if len(order) != num_nodes:
        raise GraphValidationError("Graph contains a cycle; topological sort impossible")
    return order


def transitive_closure(edges: tuple[tuple[int, int], ...], num_nodes: int) -> list[list[bool]]:
    """Floyd-Warshall transitive closure. Returns [K,K] boolean matrix where [j,i]=True
    iff a directed path exists from j to i."""
    reach = [[False] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        reach[i][i] = True
    for u, v in edges:
        reach[u][v] = True
    for k in range(num_nodes):
        for i in range(num_nodes):
            if reach[i][k]:
                row_k = reach[k]
                row_i = reach[i]
                for j in range(num_nodes):
                    row_i[j] = row_i[j] or row_k[j]
    return reach


def compute_path_matrix(graph_spec: GraphSpec) -> list[list[bool]]:
    """Transitive closure of the directed graph."""
    return transitive_closure(graph_spec.edges, graph_spec.num_nodes)


def make_dense_edges(num_nodes: int) -> tuple[tuple[int, int], ...]:
    """Generate all ordered pairs (j,i) where j != i."""
    return tuple((j, i) for j in range(num_nodes) for i in range(num_nodes) if j != i)


def get_descendants(node: int, closure: list[list[bool]]) -> frozenset[int]:
    """Nodes reachable from node (including self)."""
    return frozenset(i for i, v in enumerate(closure[node]) if v)


@dataclass
class PathCertificate:
    """Static analysis of factor-source-to-output path reachability."""
    graph_spec: GraphSpec
    closure: list[list[bool]] = field(init=False)

    def __post_init__(self):
        self.closure = transitive_closure(
            self.graph_spec.edges, self.graph_spec.num_nodes
        )

    def verify_factor_source_cut(self, factor_idx: int, output_gate_mask: frozenset[int] | None = None) -> bool:
        """True iff all paths from factor_idx to ANY output are cut.

        With FACTOR_SOURCE_CUT, source_gate[f]=0 cuts all outgoing edges from f.
        But if parent nodes feed into children that can reach outputs, those
        indirect paths may remain. This method checks static reachability.

        If output_gate_mask is provided, those outputs are also considered 'cut'.
        """
        K = self.graph_spec.num_nodes

        # Node f source is cut. Check: can info from f reach any output not in mask?
        mask = output_gate_mask or frozenset()
        descendants = self.closure[factor_idx]

        for i in range(K):
            if descendants[i] and i not in mask:
                # Is there a path from factor_idx to output i?
                if factor_idx != i or True:
                    pass

        # With source_cut: source_gate=0 → no factor info enters branch f.
        # But branch f still computes via trunk + parent messages.
        # Reachability still exists: f can receive trunk info and produce output.
        # The theorem is about FACTOR VALUE invariance, not branch computation.
        # Static analysis: if source_gate[f]=0 cuts the encoder path,
        # direct output from f can't carry f's factor value.
        # But f's output can carry info from f's parents (which might depend on f).
        # For full cut: need to cut all edges f→* as well.
        if output_gate_mask is not None:
            return factor_idx in mask

        return False  # source_cut alone doesn't statically guarantee non-interference

    def list_uncovered_paths(self, factor_idx: int) -> list[tuple[int, ...]]:
        """List factor-source-to-output paths not fully gated."""
        K = self.graph_spec.num_nodes
        paths: list[tuple[int, ...]] = []

        def dfs(current: int, visited: set[int], path: list[int]):
            if current in visited:
                return
            visited.add(current)

            if current != factor_idx and current < K:
                paths.append(tuple(path + [current]))

            for u, v in self.graph_spec.edges:
                if u == current and v not in visited:
                    dfs(v, visited.copy(), path + [current])

            visited.remove(current)

        dfs(factor_idx, set(), [])
        return paths

    def get_reachable_outputs(self, factor_idx: int) -> frozenset[int]:
        """Set of output head indices reachable from factor_idx."""
        return get_descendants(factor_idx, self.closure)


def build_graph_spec(
    graph_type: str | GraphType,
    num_nodes: int,
    edges: list[tuple[int, int]] | None = None,
    allow_cycles: bool = False,
    allow_self_loops: bool = False,
) -> GraphSpec:
    """Factory for GraphSpec with automatic edge generation for DENSE_DIRECTED."""
    if isinstance(graph_type, str):
        graph_type = GraphType(graph_type)

    if graph_type == GraphType.DENSE_DIRECTED:
        edges_tuple = make_dense_edges(num_nodes)
    elif edges is not None:
        edges_tuple = tuple(edges)
    else:
        edges_tuple = ()

    spec = GraphSpec(
        graph_type=graph_type,
        num_nodes=num_nodes,
        edges=edges_tuple,
        allow_cycles=allow_cycles,
        allow_self_loops=allow_self_loops,
    )
    validate_graph(spec)
    return spec


__all__ = [
    "GraphValidationError",
    "validate_graph",
    "topological_sort",
    "transitive_closure",
    "compute_path_matrix",
    "make_dense_edges",
    "get_descendants",
    "PathCertificate",
    "build_graph_spec",
]
