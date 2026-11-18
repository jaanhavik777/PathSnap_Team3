"""
Road network graph representation using NetworkX MultiDiGraph for PathSnap.

Preserves:
  - Directed connectivity
  - Distinct arc IDs (including parallel arcs between same endpoints)
  - Geographic arc lengths in meters (geodesic Haversine)
  - Fast shortest path caching and candidate-to-candidate travel distance
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

try:
    from src.data.spatial import haversine_distance, calculate_bearing
except ImportError:
    from .spatial import haversine_distance, calculate_bearing


class RoadGraph:
    """
    Trip-local road network graph.
    Nodes are indexed 0 to M-1.
    Arcs are indexed 0 to K-1 (matching the .arcs row index).
    """

    def __init__(
        self,
        node_coords: np.ndarray,  # shape (M, 2) [lon, lat]
        arc_nodes: np.ndarray,    # shape (K, 2) [from_node, to_node]
        trip_id: Optional[str] = None,
    ):
        self.trip_id = trip_id
        self.num_nodes = len(node_coords)
        self.num_arcs = len(arc_nodes)

        # Store arrays
        self.node_coords = np.ascontiguousarray(node_coords, dtype=np.float64)
        self.arc_nodes = np.ascontiguousarray(arc_nodes, dtype=np.int64)

        # Validate node indices
        if self.num_arcs > 0:
            max_from = int(np.max(self.arc_nodes[:, 0]))
            max_to = int(np.max(self.arc_nodes[:, 1]))
            min_from = int(np.min(self.arc_nodes[:, 0]))
            min_to = int(np.min(self.arc_nodes[:, 1]))
            if min_from < 0 or min_to < 0 or max_from >= self.num_nodes or max_to >= self.num_nodes:
                raise ValueError(
                    f"Arc node references out of range: nodes count={self.num_nodes}, "
                    f"min/max from=({min_from}, {max_from}), min/max to=({min_to}, {max_to})"
                )

        # Compute geodesic length and bearing for every arc
        self.arc_lengths = np.zeros(self.num_arcs, dtype=np.float64)
        self.arc_bearings = np.zeros(self.num_arcs, dtype=np.float64)

        # Build MultiDiGraph
        self.graph = nx.MultiDiGraph()
        for u in range(self.num_nodes):
            self.graph.add_node(u, lon=self.node_coords[u, 0], lat=self.node_coords[u, 1])

        # Map to store which arc connects u -> v: (u, v) -> list of arc_ids
        self._uv_to_arcs: Dict[Tuple[int, int], List[int]] = {}

        for arc_id in range(self.num_arcs):
            u = int(self.arc_nodes[arc_id, 0])
            v = int(self.arc_nodes[arc_id, 1])
            u_lon, u_lat = self.node_coords[u, 0], self.node_coords[u, 1]
            v_lon, v_lat = self.node_coords[v, 0], self.node_coords[v, 1]

            length = haversine_distance(u_lon, u_lat, v_lon, v_lat)
            effective_length = max(length, 0.1)
            bearing = calculate_bearing(u_lon, u_lat, v_lon, v_lat)

            self.arc_lengths[arc_id] = effective_length
            self.arc_bearings[arc_id] = bearing

            self.graph.add_edge(
                u,
                v,
                key=arc_id,
                arc_id=arc_id,
                weight=effective_length,
                bearing=bearing,
            )

            uv = (u, v)
            if uv not in self._uv_to_arcs:
                self._uv_to_arcs[uv] = []
            self._uv_to_arcs[uv].append(arc_id)

        # Shortest path distance cache: (u, v) -> distance in meters
        self._sp_cache: Dict[Tuple[int, int], float] = {}
        # Shortest path node sequence cache: (u, v) -> List of node IDs
        self._path_cache: Dict[Tuple[int, int], Optional[List[int]]] = {}

    def get_arc_nodes(self, arc_id: int) -> Tuple[int, int]:
        """Return (from_node_id, to_node_id) for the given arc_id."""
        if 0 <= arc_id < self.num_arcs:
            return int(self.arc_nodes[arc_id, 0]), int(self.arc_nodes[arc_id, 1])
        raise IndexError(f"arc_id {arc_id} out of bounds (0..{self.num_arcs - 1})")

    def get_arc_length(self, arc_id: int) -> float:
        """Return geodesic length of arc in meters."""
        if 0 <= arc_id < self.num_arcs:
            return float(self.arc_lengths[arc_id])
        raise IndexError(f"arc_id {arc_id} out of bounds (0..{self.num_arcs - 1})")

    def get_arc_bearing(self, arc_id: int) -> float:
        """Return compass bearing of arc in degrees [0, 360)."""
        if 0 <= arc_id < self.num_arcs:
            return float(self.arc_bearings[arc_id])
        raise IndexError(f"arc_id {arc_id} out of bounds (0..{self.num_arcs - 1})")

    def get_arc_coords(
        self, arc_id: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((u_lon, u_lat), (v_lon, v_lat)) for arc_id."""
        u, v = self.get_arc_nodes(arc_id)
        return (
            (float(self.node_coords[u, 0]), float(self.node_coords[u, 1])),
            (float(self.node_coords[v, 0]), float(self.node_coords[v, 1])),
        )

    def shortest_path_distance(
        self, u: int, v: int, cutoff: Optional[float] = None
    ) -> float:
        """
        Compute directed shortest path distance in meters from node u to node v.
        Returns float('inf') if no directed path exists or if distance exceeds cutoff.
        """
        if u == v:
            return 0.0

        key = (u, v)
        if key in self._sp_cache:
            return self._sp_cache[key]

        try:
            dist = nx.bidirectional_dijkstra(self.graph, u, v, weight="weight")[0]
            if cutoff is not None and dist > cutoff:
                self._sp_cache[key] = float("inf")
                return float("inf")
            self._sp_cache[key] = float(dist)
            return float(dist)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._sp_cache[key] = float("inf")
            return float("inf")

    def get_shortest_path_nodes(self, u: int, v: int) -> Optional[List[int]]:
        """Return sequence of nodes on the shortest path from u to v."""
        if u == v:
            return [u]

        key = (u, v)
        if key in self._path_cache:
            return self._path_cache[key]

        try:
            _, path = nx.bidirectional_dijkstra(self.graph, u, v, weight="weight")
            self._path_cache[key] = path
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._path_cache[key] = None
            return None

    def get_shortest_path_arcs(self, u: int, v: int) -> Optional[List[int]]:
        """
        Find shortest path from node u to node v and return corresponding list of arc IDs.
        For parallel arcs, selects the arc with the minimum length.
        """
        if u == v:
            return []

        path_nodes = self.get_shortest_path_nodes(u, v)
        if path_nodes is None or len(path_nodes) < 2:
            return None

        arcs: List[int] = []
        for i in range(len(path_nodes) - 1):
            curr_u = path_nodes[i]
            curr_v = path_nodes[i + 1]
            candidates = self._uv_to_arcs.get((curr_u, curr_v), [])
            if not candidates:
                return None
            best_arc = min(candidates, key=lambda aid: self.arc_lengths[aid])
            arcs.append(best_arc)

        return arcs

    def candidate_travel_distance(
        self,
        arc_id_1: int,
        frac_1: float,
        arc_id_2: int,
        frac_2: float,
        max_dist_cutoff: Optional[float] = 2000.0,
    ) -> float:
        """
        Compute physically valid directed road network travel distance in meters
        from candidate 1 (on arc_id_1 at fraction frac_1) to
        candidate 2 (on arc_id_2 at fraction frac_2).

        Returns float('inf') if no directed path exists or if it exceeds max_dist_cutoff.
        """
        frac_1 = max(0.0, min(1.0, frac_1))
        frac_2 = max(0.0, min(1.0, frac_2))

        # Same arc case
        if arc_id_1 == arc_id_2:
            if frac_2 >= frac_1:
                return float((frac_2 - frac_1) * self.arc_lengths[arc_id_1])
            else:
                u, v = self.get_arc_nodes(arc_id_1)
                loop_dist = self.shortest_path_distance(v, u, cutoff=max_dist_cutoff)
                if np.isinf(loop_dist):
                    return float("inf")
                total = (
                    (1.0 - frac_1) * self.arc_lengths[arc_id_1]
                    + loop_dist
                    + frac_2 * self.arc_lengths[arc_id_1]
                )
                return float(total)

        # Different arcs case
        u1, v1 = self.get_arc_nodes(arc_id_1)
        u2, v2 = self.get_arc_nodes(arc_id_2)

        remaining_arc1 = (1.0 - frac_1) * self.arc_lengths[arc_id_1]
        initial_arc2 = frac_2 * self.arc_lengths[arc_id_2]

        if v1 == u2:
            return float(remaining_arc1 + initial_arc2)

        inter_dist = self.shortest_path_distance(v1, u2, cutoff=max_dist_cutoff)
        if np.isinf(inter_dist):
            return float("inf")

        return float(remaining_arc1 + inter_dist + initial_arc2)

    def is_connected_pair(self, arc_id_1: int, arc_id_2: int) -> bool:
        """Returns True if arc_id_1 directly connects to arc_id_2 (v1 == u2) or same arc."""
        if arc_id_1 == arc_id_2:
            return True
        _, v1 = self.get_arc_nodes(arc_id_1)
        u2, _ = self.get_arc_nodes(arc_id_2)
        return v1 == u2
