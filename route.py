from __init__ import *

class Route:
    def __init__(self, id, node_u, node_v):
        self.route_id = id
        self.node_u = node_u
        self.node_v = node_v
        self.node_pair = (node_u, node_v)
        self.node_pair_id = (node_u.node_id, node_v.node_id)
        self.distance = np.random.rand() * np.random.randint(2, 8) * 1000

    def __repr__(self):
        return f"Route {self.route_id}: {self.node_u.node_id} <-> {self.node_v.node_id}"