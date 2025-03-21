import numpy as np
from node import Node

class Route:
    """
    This represents a single edge in a topology containing a tuple of orgin destination pair

    """
    def __init__(self, 
                 id: int, 
                 node_u: Node, 
                 node_v: Node,
                 min_distance: float = 2500,
                 max_distance: float = 10000
                 ) -> None:
        """
        The distance between neighbouring nodes will be calculated using the uniform proability distribution given the 
        `min_distance`, `max_distance`

        Argument:
        --------
        `id`: is the id of the route as defined in the `Topology.topology : nx.Graph`
        `node_u` and `node_v`: are the `Node`s at ends
        `min_distance`: is the minumum distance between neighbouring nodes
        `max_distance`: is the maximum distance between neighbouring nodes
        """
        self.route_id = id
        self.node_u = node_u
        self.node_v = node_v
        self.node_pair = (node_u, node_v)
        self.node_pair_id = (node_u.node_id, node_v.node_id)
        self.distance: float = np.random.choice(np.arange(min_distance, max_distance))

    def __repr__(self) -> None:
        """
        Override the to_string functionality
        """
        return f"Route {self.route_id}: {self.node_u.node_id} <-> {self.node_v.node_id}"