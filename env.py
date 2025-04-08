import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transit_system import TransitSystem
import torch
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import subgraph
from torch_geometric.data import Data



class TransitNetworkEnv(gym.Env):
    def __init__(self, is_training=True, seed=0):
        np.random.seed(seed)
        self.is_training = is_training
        with open("transit_system_config.json", "r") as file:
            self.transit_system_config = json.load(file)

        self.hours_of_opperation_per_day = self.transit_system_config["hours_of_opperation_per_day"]
        self.analysis_period_sec = self.transit_system_config["analysis_period_sec"]
        self.analysis_period_days = self.transit_system_config["analysis_period_days"]

        self.current_day = 0
        self.current_time = 0
        
        self.edge_data = {}
        self.nodes_in_routes = {}

        self.alpha_1 = 0.1
        self.alpha_2 = 0.1

        self.max_nodes = 512
        self.max_edges = self.max_nodes * 4
        self.num_node_features = 17
        self.max_routes = 32
        self.max_exit_nodes_per_route = 2

        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, self.num_node_features), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=np.inf, shape=(2, self.max_edges), dtype=np.int64),
            "edge_attr": spaces.Box(low=0, high=np.inf, shape=(self.max_edges, 1), dtype=np.float32),
        })

        self.action_space = spaces.MultiBinary(self.max_routes * self.max_exit_nodes_per_route)

    
    def reset(self, hard_reset=True, *args, **kwargs):
        if hard_reset:
            if self.is_training:
                self.seed = np.random.randint(0, self.transit_system_config["max_training_seed"])
            else:
                self.seed =  np.random.randint(self.transit_system_config["max_training_seed"], self.transit_system_config["max_training_seed"] + 
                                            self.transit_system_config["max_number_of_testing_seeds"])
            self.del_data()

        self.transit_system = TransitSystem(**self.transit_system_config, seed=self.seed) 
        self.num_nodes = len(self.transit_system.topology.nodes)
        self.num_edges = len(self.transit_system.topology.routes)
        self.num_routes = self.transit_system.topology.num_routes
        self.max_num_buses = self.num_routes * 50

        return self.get_graph(), {}
    

    def get_updated_node_data(self):
        data = []

        buses_data = {}
        for route_id in self.nodes_in_routes.keys():
            buses_data[route_id] = (len([bus for bus in self.transit_system.buses if bus.service_route == route_id and not bus.reversed]), 
                                    len([bus for bus in self.transit_system.buses if bus.service_route == route_id and bus.reversed]))

        for node in self.transit_system.topology.nodes:
            x = node.get_array()
            if node.associated_route != -1:
                x = np.append(x, buses_data[node.associated_route][0])
                x = np.append(x, buses_data[node.associated_route][1])
            else:
                x = np.append(x, -1)
                x = np.append(x, -1)
            data.append(x.astype(np.float32))

        return data

    def get_graph(self):
        graph = self.transit_system.topology.topology.copy()

        routes = self.transit_system.topology.routes
        if len(self.edge_data) == 0:
            for edge in graph.edges():
                for route in routes:
                    if route.node_pair_id == edge or route.node_pair_id[::-1] == edge:
                        self.edge_data[edge] = {"edge_attr": route.distance}
                        break

        nx.set_edge_attributes(graph, self.edge_data)
        
        if len(self.nodes_in_routes) == 0:
            self.nodes_in_routes = {k:set() for k in sorted(self.transit_system.topology.route_attributes.keys())}
            for route in routes:
                self.nodes_in_routes[route.route_id].add(route.node_u.node_id)
                self.nodes_in_routes[route.route_id].add(route.node_v.node_id)
            self.nodes_in_routes = {k: list(self.nodes_in_routes[k]) for k in self.nodes_in_routes.keys()}
        
        self.graph: nx.Graph = graph

        data = self.get_updated_node_data()
        obs = from_networkx(graph)
        obs.x = torch.from_numpy(np.stack(data, axis=0))
        self.node_indices = {v: k for k, v in enumerate(self.graph.nodes)}
        return self.fix_obs_shape(obs)

    def del_data(self):
        self.edge_data = {}
        self.nodes_in_routes = {}

    def update_graph(self):
        data = self.get_updated_node_data()
        obs = from_networkx(self.graph)
        obs.x = torch.from_numpy(np.stack(data, axis=0))
        return self.fix_obs_shape(obs)

    def fix_obs_shape(self, obs: Data):
        if obs.edge_attr.ndim == 1:
            obs.edge_attr = obs.edge_attr.unsqueeze(1)

        if obs.x.ndim > 2:
            N = obs.x.shape[0]
            x = []
            edge_attr = []
            edge_index = []
            for i in range(N):
                x.append(torch.nn.functional.pad(obs.x[i], (0, 0, 0, self.max_nodes - obs.x[i].shape[0]), mode='constant', value=0))
                edge_index.append(torch.nn.functional.pad(obs.edge_index[i], (0, self.max_edges - obs.edge_index[i].shape[1], 0, 0), mode='constant', value=0))
                edge_attr.append(torch.nn.functional.pad(obs.edge_attr[i], (0, 0, 0, self.max_edges - obs.edge_attr[i].shape[0]), mode='constant', value=0))
            
            obs.x = torch.stack(x, dim=0)
            edge_index = torch.stack(edge_index, dim=0)
            edge_attr = torch.stack(edge_attr, dim=0)
        else:
            obs.x = torch.nn.functional.pad(obs.x, (0, 0, 0, self.max_nodes - obs.x.shape[0]), mode='constant', value=0)
            obs.edge_index = torch.nn.functional.pad(obs.edge_index, (0, self.max_edges - obs.edge_index.shape[1], 0, 0), mode='constant', value=0)
            obs.edge_attr = torch.nn.functional.pad(obs.edge_attr, (0, 0, 0, self.max_edges - obs.edge_attr.shape[0]), mode='constant', value=0)        
        
        return obs
        

    def step(self, action:np.ndarray):
        """
        Arguments:
        ---------
        `action`: contains the array of actions for each route id. Each array of the action consists of a list of two binary variables.
        these binary variables corresponds to each exit node of the route and the binary variable is to indicate whether to add a bus for that exit node or not.
        Since, a single model is used for all routes, The len of action can be changed from toplogy to toplogy but the mechanism will not fail.
        """
        action = action[:self.num_routes*2]

        for i, decision in enumerate(action):
            if decision == 1:
                self.transit_system.add_bus_on_route(i // 2, reversed = False if i % 2 == 0 else True)

        reward = self.reward()
        obs: Data = self.update_graph()
        
        truncated = False
        terminated = False
        current_time = self.current_time + self.analysis_period_sec
        self.transit_system.step(current_time)
        if current_time >= self.hours_of_opperation_per_day * 3600:
            self.current_day += 1
            current_time = 0
            obs = self.reset(hard_reset=False)
            truncated = False
        
        if self.current_day >= self.analysis_period_days:
            truncated = True

        info  = {}

        if len(self.transit_system.buses) > self.max_num_buses:
            terminated = True
            reward = -1000

        return obs, reward, terminated, truncated, info

    def get_sub_graphs(self, obs: Data) -> list[Data]:
        if obs["edge_index"].ndim == 2:
            subgraphs = []
            for route_id in self.nodes_in_routes.keys():
                indices = torch.tensor([self.node_indices[i] for i in self.nodes_in_routes[route_id]]).long()
                edge_index, edge_attr = subgraph(indices, obs["edge_index"].long(), obs["edge_attr"], relabel_nodes=False)
                sub_data = Data(
                    x=obs["x"][indices],
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=obs["y"][indices] if obs["y"] is not None else None
                )
                subgraphs.append(self.fix_obs_shape(sub_data))
        else:
            subgraphs = []
            for route_id in self.nodes_in_routes.keys():
                indices = torch.tensor([self.node_indices[i] for i in self.nodes_in_routes[route_id]]).long()
                batch_size = obs["edge_index"].shape[0]
                sub_datas = []
                for i in range(batch_size):
                    edge_index, edge_attr = subgraph(indices, obs["edge_index"][i].long(), obs["edge_attr"][i], relabel_nodes=False)
                    sub_data = Data(
                        x=obs["x"][i][indices],
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=None if "y" not in obs else obs["y"][i][indices]
                    )
                    sub_datas.append(self.fix_obs_shape(sub_data))

                sub_data = Data(x=torch.stack([sub_data.x for sub_data in sub_datas], dim=0),
                                edge_index=torch.stack([sub_data.edge_index for sub_data in sub_datas], dim=0),
                                edge_attr=torch.stack([sub_data.edge_attr for sub_data in sub_datas], dim=0),
                                y=None if "y" not in obs else torch.stack([sub_data.y for sub_data in sub_datas], dim=0))

                subgraphs.append(sub_data)

        return subgraphs

    def reward(self) -> float:
        """
        Reward function here is inspired from the https://arxiv.org/abs/2107.07066
        """

        # r = 1 - om/em  if action==0 else om/em (where om/em is demand / bus capacity) (Also penalize the action=0 for waiting time and both actions for stranding time)  Guanqun Ai
        # since we deal with an for each exit node seperately we will not use the above function directly

        reward = 0

        # First lets think from the rpespective of passengers
        total_capacity = sum([bus.capacity - len(bus.passengers) for bus in self.transit_system.buses]) + 1
        total_demand = sum([len(node.passengers) for node in self.transit_system.topology.nodes])

        demand_capacity_ratio = total_demand / total_capacity
        reward += 1 - demand_capacity_ratio # forcing the model to incease the number of buses to achive highest rewards


        num_passengers = [len(node.passengers) for node in self.transit_system.topology.nodes]
        avg_waiting_time = [np.median([passenger.waiting_time for passenger in node.passengers]) for node, num_passenger in zip(self.transit_system.topology.nodes, num_passengers) if num_passenger > 0]
        if len(avg_waiting_time) > 0:
            avg_waiting_time = np.mean(avg_waiting_time)/3600 #hours
        else:
            avg_waiting_time = 0
        
        avg_stranding_count = [np.median([passenger.stranding_counts for passenger in node.passengers]) for node, num_passenger in zip(self.transit_system.topology.nodes, num_passengers) if num_passenger > 0]
        if len(avg_stranding_count) > 0:
            avg_stranding_count = np.mean(avg_stranding_count)
        else:
            avg_stranding_count = 0 #counts

        reward += - self.alpha_1 * avg_waiting_time - self.alpha_2 * avg_stranding_count #penalizing the long waiting time and standing counts
        
        # Now lets think from the prespectives of opperators

        operator_cost = 0
        for route_id in self.transit_system.topology.route_attributes.keys():
            num_buses = sum([1 for bus in self.transit_system.buses if bus.service_route == route_id])
            operator_cost = num_buses * self.transit_system.topology.route_attributes[route_id]["percent_length"]

        reward += -operator_cost
        
        return reward

    def render(self):
        pass