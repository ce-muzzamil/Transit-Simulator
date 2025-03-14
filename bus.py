from __init__ import *

class Bus:
    def __init__(self, capacity, 
                 avg_speed, 
                 service_route,
                 step_interval,
                 topology,
                 reversed,
                 ):
        self.capacity = capacity
        self.avg_speed = avg_speed
        self.speed = avg_speed #kmph
        self.service_route = service_route
        self.location = 0.0
        self.step_interval = step_interval
        self.reversed = reversed

        self.topology = topology
        nodes_ids = [[route.node_u.node_id, route.node_v.node_id] for route in self.topology.routes if route.route_id == self.service_route]
        nodes_ids = list(set(sum(nodes_ids, [])))
        self.num_stations_in_trajectory = len(nodes_ids)

        subgraph = self.topology.topology.subgraph(nodes_ids)
        self.neighbors = {node: list(nx.neighbors(subgraph, node)) for node in nodes_ids}
        self.exit_nodes = [node_id for node_id in nodes_ids if len(self.neighbors[node_id]) == 1]
        self.routes = [route for route in self.topology.routes if route.route_id == self.service_route]

        self.passengers = []
        self.initilize_trajectory()
        
    def initilize_trajectory(self):
        if self.reversed:
            self.routes = self.routes[::-1]
            current_node_id = min(self.exit_nodes)
        else:
            current_node_id = max(self.exit_nodes)
        self.current_node = self.get_node_by_id(current_node_id)

        node_u = self.current_node
        self.to_go = [node_u]
        for _ in range(self.num_stations_in_trajectory):
            next_node_ids = self.neighbors[node_u.node_id]
            for node_id in next_node_ids:
                next_node = self.get_node_by_id(node_id)
                if next_node not in self.to_go:
                    self.to_go.append(next_node)
                    node_u = next_node
                    break

        self.distances = []
        for node_pair in zip(self.to_go[:-1], self.to_go[1:]):
            for route in self.routes:
                if  all([node in route.node_pair for node in node_pair]):
                    self.distances.append(route.distance)
                    break
        
        self.distance_next_node = self.distances.pop(0)

    def get_node_by_id(self, id):
        for node in self.topology.nodes:
            if node.node_id == id:
                return node
            
    def step(self, time):
        self.speed = max(5, self.avg_speed + np.random.normal(loc=0, scale=20))
        self.distance_next_node -= self.speed * self.step_interval
        if self.distance_next_node <= 0:

            current_node = self.to_go.pop(0)
            current_node.bus_arrived(time, self)
            if len(self.distances)>0:
                distance_next_node = self.distances.pop(0)
                self.distance_next_node = distance_next_node - abs(self.distance_next_node)
            else:
                self.distance_next_node = -np.inf

        if len(self.to_go) == 0:
            self.reversed = not self.reversed
            self.initilize_trajectory()