from __init__ import *

class Route:
    """
    This represents a single edge in a topology containing a tuple of orgin destination pair

    """
    def __init__(self, id: int, node_u: Node, node_v: Node) -> None:
        """
        Argument:
        --------
        `id`: is the id of the route as defined in the `Topology.topology : nx.Graph`
        `node_u` and `node_v`: are the `Node`s at ends
        """
        self.route_id = id
        self.node_u = node_u
        self.node_v = node_v
        self.node_pair = (node_u, node_v)
        self.node_pair_id = (node_u.node_id, node_v.node_id)
        self.distance = np.random.rand() * np.random.randint(2, 8) * 1000

    def __repr__(self) -> None:
        """
        Override the to_string functionality
        """
        return f"Route {self.route_id}: {self.node_u.node_id} <-> {self.node_v.node_id}"