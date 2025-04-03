import json
import numpy as np
import gymnasium as gym
from transit_system import TransitSystem
import networkx as nx
from torch_geometric.utils.convert import from_networkx

class TransitNetworkEnv(gym.Env):
    def __init__(self, is_training=True):
        self.is_training = is_training
        with open("transit_system_config.json", "r") as file:
            self.transit_system_config = json.load(file)

        self.hours_of_opperation_per_day = self.transit_system_config["hours_of_opperation_per_day"]
        self.analysis_period_sec = self.transit_system_config["analysis_period_sec"]
        self.analysis_period_days = self.transit_system_config["analysis_period_days"]

        self.current_day = 0
        self.current_time = 0
        
        self.edge_data = {}
        
    def reset(self, hard_reset=False):
        if hard_reset:
            if self.is_training:
                self.seed = np.random.randint(0, self.transit_system_config["max_training_seed"])
            else:
                self.seed =  np.random.randint(self.transit_system_config["max_training_seed"], self.transit_system_config["max_training_seed"] + 
                                            self.transit_system_config["max_number_of_testing_seeds"])
                
        self.transit_system = TransitSystem(**self.transit_system_config, seed=self.seed)
        graph, sub_graphs = self.get_graph()

        return (graph, sub_graphs)

    def get_graph(self):
        graph = self.transit_system.topology.topology.copy()

        routes = self.transit_system.topology.routes
        if len(self.edge_data) == 0:
            for edge in graph.edges():
                for route in routes:
                    if route.node_pair_id == edge or route.node_pair_id[::-1] == edge:
                        self.edge_data[edge] = {"distance": route.distance}
                        break

        graph = nx.set_edge_attributes(graph, self.edge_data)

        nodes = self.transit_system.topology.nodes
        nodes_in_routes = {set() for _ in set([route.route_id for route in routes])}
        
        data = {}
        for node in nodes:
            data[node.node_id] = node.get_dct()
            for route_id in nodes_in_routes.keys():
                if node.node_id in nodes_in_routes[route_id]:
                    data[node.node_id]["num_buses_on_route"] = len([bus for bus in self.transit_system.buses if bus.service_route == route_id])

        graph = nx.set_node_attributes(graph, data)

        for route in routes:
            nodes_in_routes[route.route_id].add(route.node_u.node_id)
            nodes_in_routes[route.route_id].add(route.node_v.node_id)

        nodes_in_routes = {k: list(nodes_in_routes[k]) for k in nodes_in_routes.keys()}
        
        return from_networkx(graph), [from_networkx(nx.subgraph(graph, nodes_in_routes[k])) for k in sorted(nodes_in_routes.keys())]


    def step(self, action:list[list[int]]):
        """
        Arguments:
        ---------
        `action`: contains the array of actions for each route id. Each array of the action consists of a list of two binary variables.
        these binary variables corresponds to each exit node of the route and the binary variable is to indicate whether to add a bus for that exit node or not.
        Since, a single model is used for all routes, The len of action can be changed from toplogy to toplogy but the mechanism will not fail.
        """
        for i, decisions in enumerate(action):
            for j, decision in enumerate(decisions):
                reversed  = 0 if j == 0 else 1
                if decision == 1:
                    self.transit_system.add_bus_on_route(i, reversed=reversed)

        reward = self.reward()
        obs = self.get_graph()
        
        current_time = self.current_time + self.analysis_period_sec
        if current_time >= self.hours_of_opperation_per_day * 3600:
            self.current_day += 1
            current_time = 0
            obs = self.reset(hard_reset=False)
            truncated = False
        
        if self.current_day >= self.analysis_period_days:
            obs = self.reset(hard_reset=True)
            truncated = True

        info  = {}

        return obs, reward, False, truncated, info
    


    def reward(self):
        total_distance_traversed = sum([bus.total_distance_traversed/self.transit_system_config["max_distance"] for bus in self.transit_system.buses])
        total_waiting_time = sum([sum([passenger.waiting_time/(self.hours_of_opperation_per_day * 3600) for passenger in node.passengers]) for node in self.transit_system.topology.nodes])
        total_stranding_counts = sum([sum([passenger.stranding_counts/10 for passenger in node.passengers]) for node in self.transit_system.topology.nodes])
        
        return - total_distance_traversed - total_waiting_time - total_stranding_counts

    def render(self):
        pass