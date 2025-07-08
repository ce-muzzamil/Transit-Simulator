import numpy as np
import pandas as pd
from node import Node
from route import Route
from matplotlib import pyplot as plt
import networkx as nx
from functions import softmax


class Topology:
    """
    This class represents the transit routes for a transit system. It has the following features.

    * It generates a completely different topology for a given `seed` with different `Node`s and node attributes
    * Given the `Node` attributes It can produce the OD matrix for a given time
    """

    def __init__(
        self,
        min_num_stops_per_route: int = 8,
        max_num_stops_per_route: int = 32,
        min_num_route_per_toplogy: int = 4,
        max_num_route_per_toplogy: int = 12,
        hours_of_opperation_per_day: int = 18,
        analysis_period_sec: int = 60,
        mean_population_density: float = 300.0,
        std_population_density: float = 200.0,
        min_population_density: float = 100.0,
        mean_catchment_radius: float = 2.0,
        std_catchment_radius: float = 1.0,
        min_catchment_radius: float = 0.5,
        min_transit_users_proportion: float = 0.05,
        max_transit_users_proportion: float = 0.3,
        min_distance: float = 2500,
        max_distance: float = 10000,
        seed: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """
        Argument:
        --------
        `[min,max]_num_stops_per_route` : is the upper and lower bound of nodes for each route
        `[min,max]_num_route_per_toplogy` : is the upper and lower bound of routes for each topology
        `hours_of_opperation_per_day` : is the maximum hours the buses will keep running
        `analysis_period_sec` : is the least count of time
        `mean_population_density`: is the mean population density of catchment area of each station
        `std_population_density`: is the standrd deviation of population density of catchment area of each station
        `min_population_density`: is the lowerbound to clip smaller values for the population density of catchment area of each station
        `mean_catchment_radius`: is the mean area of the catchment area of each station
        `std_catchment_radius`: is the standard deviation area of the catchment area of each station
        `min_catchment_radius`: is the lowerbound to clip smaller values for the area of catchment area of each station
        `min_transit_users_proportion`: is the minimum ratio of transit users given the population for a station
        `max_transit_users_proportion`: is the maximum ration of transit users given the population for a station
        `min_distance`: is the minumum distance between neighbouring nodes
        `max_distance`: is the maximum distance between neighbouring nodes
        """
        self.min_num_stops_per_route = min_num_stops_per_route
        self.max_num_stops_per_route = max_num_stops_per_route
        self.min_num_route_per_toplogy = min_num_route_per_toplogy
        self.max_num_route_per_toplogy = max_num_route_per_toplogy
        self.hours_of_opperation_per_day = hours_of_opperation_per_day
        self.analysis_period_sec = analysis_period_sec

        self.mean_population_density = mean_population_density
        self.std_population_density = std_population_density
        self.min_population_density = min_population_density
        self.mean_catchment_radius = mean_catchment_radius
        self.std_catchment_radius = std_catchment_radius
        self.min_catchment_radius = min_catchment_radius
        self.min_transit_users_proportion = min_transit_users_proportion
        self.max_transit_users_proportion = max_transit_users_proportion
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.seed = seed

        self.generate_nodes()
        self.generate_routes()
        self.brush()
        self.generate_od_routes()
        self.initiallize_traffic_data()

        self.route_ids = sorted(
            set([d["label"] for _, _, d in self.topology.edges(data=True)])
        )
        self.route_attributes = {}
        for route_id in self.route_ids:
            self.route_attributes[route_id] = {
                "distance": sum(
                    [
                        route.distance
                        for route in self.routes
                        if route.route_id == route_id
                    ]
                )
            }

        for route_id in self.route_ids:
            self.route_attributes[route_id]["percent_length"] = self.route_attributes[
                route_id
            ]["distance"] / sum(
                [
                    self.route_attributes[route_id]["distance"]
                    for route_id in self.route_ids
                ]
            )

        for node in self.nodes:
            if not node.is_transfer:
                for route in self.routes:
                    if node in route.node_pair:
                        node.associated_route = route.route_id
                        break

        for node in self.nodes:
            node.affliated_route_ids = {
                route.route_id for route in node.affiliated_routes
            }

        self.num_routes = len(self.route_ids)
        node_indices = {node.node_id: index for index, node in enumerate(self.nodes)}
        self.node_nbr_indices = {
            node_id: [node_indices[nbr_id] for nbr_id in self.neighbors[node_id]]
            for node_id in node_indices.keys()
        }

    def fix_route_clusters(self) -> None:
        """
        Fixes if there is discontinuity in the topology. It makes sure all the nodes are accessible to each other
        """
        nodes = {}
        for u, _, data in self.topology.edges(data=True):
            if data["label"] not in nodes:
                nodes[data["label"]] = []
            nodes[data["label"]].append(u)

        for route_id in nodes.keys():
            rnodes = nodes[route_id]
            subgraph: nx.Graph = nx.subgraph(self.topology, rnodes)
            subgraph = nx.Graph(subgraph)
            to_drop = []
            for u, v, data in subgraph.edges(data=True):
                if data["label"] != route_id:
                    to_drop.append((u, v))
                    to_drop.append((v, u))

            subgraph.remove_edges_from(to_drop)
            components = list(nx.connected_components(subgraph))

            if len(components) > 1:
                all_exit_nodes = []
                for component in components:
                    rsubgraph: nx.Graph = nx.subgraph(subgraph, component)
                    rsubgraph = nx.Graph(rsubgraph)
                    neighbors = {
                        node: list(nx.neighbors(rsubgraph, node)) for node in component
                    }

                    exit_nodes = [
                        node_id for node_id in component if len(neighbors[node_id]) == 1
                    ]
                    all_exit_nodes.append(exit_nodes)

                for i in range(len(all_exit_nodes) - 1):
                    is_connected = False
                    for j in range(i + 1, len(all_exit_nodes)):
                        if is_connected:
                            break

                        for u in all_exit_nodes[i]:
                            if is_connected:
                                break

                            for v in all_exit_nodes[j]:
                                if not self.topology.has_edge(u, v):
                                    self.topology.add_edge(u, v, label=route_id)
                                    is_connected = True
                                    break

        route_clusters = {}
        for route_id_1 in nodes.keys():
            for route_id_2 in nodes.keys():
                if route_id_1 != route_id_2:
                    if nx.has_path(
                        self.topology, nodes[route_id_1][0], nodes[route_id_2][0]
                    ):
                        if route_id_1 not in route_clusters:
                            route_clusters[route_id_1] = [route_id_1]
                        route_clusters[route_id_1].append(route_id_2)

            if (
                route_id_1 in route_clusters
                and np.isin(list(nodes.keys()), route_clusters[route_id_1]).all()
            ):
                return

        def sort_and_tuple(x):
            return tuple(sorted(x))

        route_clusters = list(set(map(sort_and_tuple, route_clusters.values())))

        for c1, c2 in zip(route_clusters[:-1], route_clusters[1:]):
            r1 = np.random.choice(c1)
            r2 = np.random.choice(c2)

            r1_nodes = sorted(
                set(
                    sum(
                        [
                            [u, v]
                            for u, v, data in self.topology.edges(data=True)
                            if data["label"] == r1
                        ],
                        [],
                    )
                )
            )
            r2_nodes = sorted(
                set(
                    sum(
                        [
                            [u, v]
                            for u, v, data in self.topology.edges(data=True)
                            if data["label"] == r2
                        ],
                        [],
                    )
                )
            )

            u = np.random.choice(r1_nodes)
            v = np.random.choice(r2_nodes)
            edges = [
                (uu, vv, data)
                for uu, vv, data in self.topology.edges(data=True)
                if u in (uu, vv)
            ]
            self.topology.remove_node(u)
            for uu, vv, data in edges:
                if uu == u:
                    self.topology.add_edge(max(v, vv), min(v, vv), label=data["label"])
                else:
                    self.topology.add_edge(max(uu, v), min(uu, v), label=data["label"])

    def check_if_interval(self, time: int, interval: list) -> bool:
        """
        Argument:
        --------
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation
        `interval` is a list of fixed timestamps

        Return:
        ------
        returns if the `time` belongs to the `interval`
        """
        return interval[0] <= time < interval[1]

    def get_od_mat_for_time(self, time: int) -> np.ndarray:
        """
        Argument:
        --------
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation

        Return:
        ------
        od matrix of shape (NxN) where N is the number of nodes
        """
        assert (
            time < self.hours_of_opperation_per_day * 3600
            and time > -1
            and np.ceil(time) == np.floor(time)
        )

        num_nodes = self.topology.number_of_nodes()
        od_mat = np.random.rand(num_nodes, num_nodes)
        for node in self.nodes:
            idc = self.node_nbr_indices[node.node_id]
            od_mat[idc] = od_mat[idc] ** (1 / 3)
        od_mat[range(num_nodes), range(num_nodes)] = -np.inf

        if self.check_if_interval(time, self.schools_times):
            od_mat[:, self.schools] = od_mat[:, self.schools] * 5
            od_mat[self.schools, :] = od_mat[self.schools, :] / 5

        if self.check_if_interval(time, self.offices_times):
            od_mat[:, self.offices] = od_mat[:, self.offices] * 5
            od_mat[self.offices, :] = od_mat[self.offices, :] / 5

        if self.check_if_interval(time, self.shopping_times):
            od_mat[:, self.shopping] = od_mat[:, self.shopping] * 3
            od_mat[self.shopping, :] = od_mat[self.shopping, :] / 3

        if self.check_if_interval(time, self.residential_times):
            od_mat[:, self.residentials] = od_mat[:, self.residentials] * 2
            od_mat[self.residentials, :] = od_mat[self.residentials, :] / 2

        od_mat = (
            (
                self.transit_users[:, None] * self.analysis_period_sec / 50000
            )
            * softmax(od_mat, axis=1)
            * self.traffic_curve[time]
        )

        return od_mat

    def generic_traffic_curve(self) -> np.ndarray:
        """
        This function creats a generalized traffic curve that the simulation will roughly follows

        Returns:
        -------
        traffic volume at all the values of `time`
        """

        y = np.zeros(60 * 60 * self.hours_of_opperation_per_day)
        self.hours_of_opperation_per_day
        p = [int(i*self.hours_of_opperation_per_day) for i in [0.15, 0.30, 0.45, 0.60, 0.75]]
        y[: p[0] * 3600] = np.linspace(0, 0.6, p[0] * 3600)
        y[p[0] * 3600 : p[1] * 3600] = np.linspace(0.6, 0.2, y[p[0] * 3600 : p[1] * 3600].shape[0])
        y[p[1] * 3600 : p[2] * 3600] = np.linspace(0.2, 0.5, y[p[1] * 3600 : p[2] * 3600].shape[0])
        y[p[2] * 3600 : p[3] * 3600] = np.linspace(0.5, 1.0, y[p[2] * 3600 : p[3] * 3600].shape[0])
        y[p[3] * 3600 : p[4] * 3600] = np.ones(y[p[3] * 3600 : p[4] * 3600].shape[0])
        y[p[4] * 3600 : self.hours_of_opperation_per_day * 3600] = np.linspace(
            1, 0.05, y[p[4] * 3600 : self.hours_of_opperation_per_day * 3600].shape[0]
        )

        y = pd.Series(y).rolling(3600).mean().values
        mask = np.isnan(y)
        y[mask] = np.linspace(0.1, y[~mask][0], mask.sum())

        return y

    def initiallize_traffic_data(self) -> None:
        """
        calls the `generic_traffic_curve` method with preloaded data
        """
        num_nodes = self.topology.number_of_nodes()
        nodes_list = list(range(num_nodes))
        np.random.shuffle(nodes_list)

        school_portion = int(0.2 * num_nodes)
        office_portion = int(0.2 * num_nodes)
        shopping_portion = int(0.1 * num_nodes)

        self.schools = nodes_list[:school_portion]
        self.offices = nodes_list[school_portion : school_portion + office_portion]
        self.shopping = nodes_list[
            school_portion
            + office_portion : shopping_portion
            + school_portion
            + office_portion
        ]
        self.residentials = nodes_list[
            shopping_portion + school_portion + office_portion :
        ]

        p = [int(i*self.hours_of_opperation_per_day) for i in [0.15, 0.30, 0.45, 0.60, 0.75]]

        self.offices_times = [0, p[1] * 3600]
        self.schools_times = [0 * 3600, p[1] * 3600]
        self.shopping_times = [p[2] * 3600, p[4] * 3600]
        self.residential_times = [p[2] * 3600, p[4] * 3600]

        self.traffic_curve = self.generic_traffic_curve()

    def generate_od_routes(self) -> None:
        """
        Generates dict containing the shortest path calculated using `nx.shortest_path` between all the `node_ids` of `self.topology`
        """
        self.od_routes = {}
        nodes = {node.node_id: node for node in self.nodes}
        node_ids = list(nodes.keys())

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                path = nx.shortest_path(self.topology, node_ids[i], node_ids[j])
                u, v = max(node_ids[i], node_ids[j]), min(node_ids[i], node_ids[j])
                self.od_routes[(u, v)] = [nodes[node_id] for node_id in path]
                self.od_routes[(v, u)] = [nodes[node_id] for node_id in path][::-1]

        for node_id in nodes.keys():
            for node_id_2 in nodes.keys():
                if node_id != node_id_2:
                    path = self.od_routes[(node_id, node_id_2)]
                    nodes[node_id].od_route[node_id_2] = path

                    distance = 0.0
                    for route in self.routes:
                        for u, v in zip(path[:-1], path[1:]):
                            if u in route.node_pair and v in route.node_pair:
                                distance += route.distance
                    nodes[node_id].od_distance[node_id_2] = distance

    def brush(self) -> None:
        """
        Sequentially calls essential methods to generate topology
        """
        self.check_connectivity()
        self.fix_zero_connectivity()
        self.check_connectivity()
        self.drop_redundant_routes()
        self.check_connectivity()
        self.fix_zero_connectivity()
        self.check_connectivity()
        self.get_graph()
        self.fix_route_clusters()
        self.remove_isolated_nodes()
        self.find_neighbors()
        self.fix_splinter_issue()
        self.fix_route_loop_and_discontinuity()
        self.find_neighbors()
        self.initiallize_traffic_data()
        self.process_nodes_and_routes()
    
        
    def fix_branching(self):
        route_ids = sorted(
            set([data["label"] for _, _, data in self.topology.edges(data=True)])
        )

        nodes_in_routes = {
            k: list(
                set(
                    sum(
                        [
                            [u, v]
                            for u, v, d in self.topology.edges(data=True)
                            if d["label"] == k
                        ],
                        [],
                    )
                )
            )
            for k in route_ids
        }

        to_remove = []
        for k in route_ids:
            nodes = nodes_in_routes[k]
            transfers = []
            for node in nodes:
                neighbors = self.neighbors[node]
                if len(neighbors) <= 2:
                    continue
                else:
                    transfers.append(node)
            
            for transfer in transfers:
                rids = [self.topology.get_edge_data(transfer, nbr)["label"]==k for nbr in self.neighbors[transfer]]
                if sum(rids)>2:
                    neighbors = self.neighbors[transfer]
                    for nbr in neighbors:
                        if self.topology.get_edge_data(transfer, nbr)["label"] == k:
                            if len(self.neighbors(nbr)) == 1:
                                to_remove.append(nbr)

        for node in to_remove:
            self.topology.remove_node(node)

    def fix_route_loop_and_discontinuity(self):
        route_ids = sorted(
            set([data["label"] for _, _, data in self.topology.edges(data=True)])
        )
        nodes_in_routes = {
            k: list(
                set(
                    sum(
                        [
                            [u, v]
                            for u, v, d in self.topology.edges(data=True)
                            if d["label"] == k
                        ],
                        [],
                    )
                )
            )
            for k in route_ids
        }
        for route_id in route_ids:

            nodes = nodes_in_routes[route_id]

            subgraph: nx.Graph = nx.subgraph(self.topology, nodes)
            subgraph = nx.Graph(subgraph)
            to_drop = []
            for u, v, data in subgraph.edges(data=True):
                if data["label"] != route_id:
                    to_drop.append((u, v))
                    to_drop.append((v, u))

            subgraph.remove_edges_from(to_drop)

            neighbors = {node: list(nx.neighbors(subgraph, node)) for node in nodes}
            exit_nodes = [node_id for node_id in nodes if len(neighbors[node_id]) == 1]

            if len(exit_nodes) == 0:
                all_edges_in_route_id = subgraph.edges
                self.topology.remove_edges_from(all_edges_in_route_id)
                self.topology.add_edges_from(
                    [(u, v, {"label": route_id}) for u, v in zip(nodes[:-1], nodes[1:])]
                )

            subgraph: nx.Graph = nx.subgraph(self.topology, nodes)
            subgraph = nx.Graph(subgraph)
            to_drop = []
            for u, v, data in subgraph.edges(data=True):
                if data["label"] != route_id:
                    to_drop.append((u, v))
                    to_drop.append((v, u))

            subgraph.remove_edges_from(to_drop)

            neighbors = {node: list(nx.neighbors(subgraph, node)) for node in nodes}
            exit_nodes = [node_id for node_id in nodes if len(neighbors[node_id]) == 1]

            if len(exit_nodes) > 2:
                options = []
                for u in exit_nodes:
                    for v in exit_nodes:
                        if u != v:
                            if (u, v) not in self.topology.edges and (
                                v,
                                u,
                            ) not in self.topology.edges:
                                options.append((u, v))

                selected = np.random.choice(len(options))
                u, v = options[selected]
                self.topology.add_edge(u, v, label=route_id)

            subgraph: nx.Graph = nx.subgraph(self.topology, nodes)
            subgraph = nx.Graph(subgraph)
            to_drop = []
            for u, v, data in subgraph.edges(data=True):
                if data["label"] != route_id:
                    to_drop.append((u, v))
                    to_drop.append((v, u))

            subgraph.remove_edges_from(to_drop)

            neighbors = {node: list(nx.neighbors(subgraph, node)) for node in nodes}
            looped_nodes = [
                node for node in neighbors.keys() if len(neighbors[node]) > 2
            ]

            for u in looped_nodes:
                for v in neighbors[u]:
                    temp_subgraph = subgraph.copy()

                    if u != v:
                        temp_subgraph.remove_edge(u, v)

                        if len(list(nx.connected_components(temp_subgraph))) == 1:
                            self.topology.remove_edge(u, v)

                            subgraph: nx.Graph = nx.subgraph(self.topology, nodes)
                            subgraph = nx.Graph(subgraph)
                            to_drop = []
                            for u_, v_, data in subgraph.edges(data=True):
                                if data["label"] != route_id:
                                    to_drop.append((u_, v_))
                                    to_drop.append((v_, u_))

                            subgraph.remove_edges_from(to_drop)

    def process_nodes_and_routes(self) -> None:
        """
        Updating `self.nodes` and `self.routes` using `self.topology.nodes` and `self.topology.edges`
        """
        self.nodes: list[Node] = [
            node for node in self.nodes if node.node_id in self.topology.nodes
        ]
        nodes = {node.node_id: node for node in self.nodes}
        self.routes: list[Route] = [
            Route(
                data["label"],
                nodes[u],
                nodes[v],
                min_distance=self.min_distance,
                max_distance=self.max_distance,
            )
            for u, v, data in self.topology.edges(data=True)
        ]
        self.transit_users = np.array([node.transit_users for node in self.nodes])

        for node in self.nodes:
            for node_id in self.exit_nodes:
                if node.node_id == node_id:
                    node.is_exit = True

            for node_id in self.transfer_nodes:
                if node.node_id == node_id:
                    node.is_transfer = True

            for route in self.routes:
                if node.node_id in route.node_pair_id:
                    node.affiliated_routes.add(route)

            for node_id in self.schools:
                if node.node_id == node_id:
                    node.zone_type = "school"

            for node_id in self.offices:
                if node.node_id == node_id:
                    node.zone_type = "office"

            for node_id in self.shopping:
                if node.node_id == node_id:
                    node.zone_type = "shopping"

            for node_id in self.residentials:
                if node.node_id == node_id:
                    node.zone_type = "residentials"

        route_ids = list(set([route.route_id for route in self.routes]))
        for route in route_ids:
            nodes = list(
                set(
                    sum(
                        [
                            [u, v]
                            for u, v, data in self.topology.edges(data=True)
                            if data["label"] == route
                        ],
                        [],
                    )
                )
            )

            subgraph: nx.Graph = nx.subgraph(self.topology, nodes)
            subgraph = nx.Graph(subgraph)
            to_drop = []
            for u, v, data in subgraph.edges(data=True):
                if data["label"] != route:
                    to_drop.append((u, v))
                    to_drop.append((v, u))

            subgraph.remove_edges_from(to_drop)

            neighbors = {node: list(nx.neighbors(subgraph, node)) for node in nodes}
            exit_nodes = [node_id for node_id in nodes if len(neighbors[node_id]) == 1]

            for node in self.nodes:
                if node.node_id in nodes:
                    node.exit_nodes.extend(exit_nodes)

    def get_graph(self) -> None:
        """
        Generates `nx.Graph` using existing data in `self.nodes` and `self.routes`
        """
        func = lambda route: zip(route[:-1], route[1:])
        routes = map(func, self.routes)

        self.topology = nx.Graph()
        for node in self.nodes:
            self.topology.add_node(node.node_id)
        for i, route in enumerate(routes):
            for node_pair in route:
                if node_pair[0] != node_pair[1]:
                    node_pair = max(*node_pair), min(*node_pair)
                    self.topology.add_edge(*node_pair, label=i)

    def fix_splinter_issue(self) -> None:
        """
        Fixes the issue where an exit node exist just after the transfer node to simplify the topology
        """
        tbr = []
        exit_nodes = [
            node_id for node_id, nbrs in self.neighbors.items() if len(nbrs) == 1
        ]
        for node_id, nbrs in self.neighbors.items():
            if len(nbrs) > 2:
                for nbr in self.neighbors[node_id]:
                    if nbr in exit_nodes:
                        tbr.append(nbr)

        self.topology.remove_nodes_from(tbr)
        self.remove_isolated_nodes()
        self.find_neighbors()
        self.exit_nodes = [
            node_id for node_id, nbrs in self.neighbors.items() if len(nbrs) == 1
        ]
        self.transfer_nodes = [
            node_id for node_id, nbrs in self.neighbors.items() if len(nbrs) > 2
        ]
        node_to_route = {}
        for u, v, data in self.topology.edges(data=True):
            if u not in node_to_route:
                node_to_route[u] = set()
            if v not in node_to_route:
                node_to_route[v] = set()
            node_to_route[u].add(data["label"])
            node_to_route[v].add(data["label"])

        for node_id, route_ids in node_to_route.items():
            if len(route_ids) > 1:
                self.transfer_nodes.append(node_id)
        
        self.transfer_nodes = list(set(self.transfer_nodes))


    def drop_redundant_routes(self) -> None:
        """
        Solves the triangular connections and removing loops to simplify the topology
        """
        tbr = []
        for route in self.r2r_connectivity.keys():
            if self.r2r_connectivity[route] > self.num_routes // 2:
                tbr.append(self.routes[route])

        for k in tbr:
            self.routes.remove(k)

    def find_neighbors(self) -> None:
        """
        Searching for the neighbors for each node and storing them in `self.neighbors`
        """
        self.neighbors = {}
        for node in self.topology.nodes:
            self.neighbors[node] = list(nx.neighbors(self.topology, node))

    def remove_isolated_nodes(self) -> None:
        """
        Removes nodes that are not connected to any other nodes
        """
        isolated_nodes = nx.isolates(self.topology)
        self.topology.remove_nodes_from(list(isolated_nodes))

    def generate_nodes(self) -> None:
        """
        Generates nodes using uniform probability distribution given the min and max nodes per route and min and max routes per topology
        """
        self.num_stations = np.random.randint(
            self.max_num_stops_per_route * self.min_num_route_per_toplogy,
            self.max_num_stops_per_route * self.max_num_route_per_toplogy,
        )

        self.nodes = []
        for node_id in range(self.num_stations):
            self.nodes.append(
                Node(
                    node_id=node_id,
                    mean_population_density=self.mean_population_density,
                    std_population_density=self.std_population_density,
                    min_population_density=self.min_population_density,
                    mean_catchment_radius=self.mean_catchment_radius,
                    std_catchment_radius=self.std_catchment_radius,
                    min_catchment_radius=self.min_catchment_radius,
                    min_transit_users_proportion=self.min_transit_users_proportion,
                    max_transit_users_proportion=self.max_transit_users_proportion,
                    analysis_period_sec=self.analysis_period_sec,
                )
            )

    def generate_routes(self) -> None:
        """
        Generates routes using generated nodes uisng conditionalized uniform probability distribution.
        """
        max_num_routes = min(
            self.num_stations // self.min_num_stops_per_route,
            self.max_num_route_per_toplogy,
        )
        min_num_routes = max(
            self.num_stations // self.max_num_stops_per_route,
            self.min_num_route_per_toplogy,
        )
        if min_num_routes < max_num_routes:
            self.num_routes = np.random.randint(min_num_routes, max_num_routes)
        else:
            self.num_routes = max_num_routes

        self.routes = []
        used_nodes = []
        nbrs = {k.node_id: [] for k in self.nodes}

        probability = np.array([1 / self.num_stations] * self.num_stations)
        for _ in range(self.num_routes):
            route = np.random.choice(
                self.nodes,
                np.random.randint(
                    self.min_num_stops_per_route, self.max_num_stops_per_route
                ),
                p=probability,
            )

            route = np.unique([node.node_id for node in route]).tolist()
            _route = [route[0]]
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if not v in nbrs[u] and v != u:
                    nbrs[u].append(v)
                    _route.append(v)

            route = _route
            used_nodes.extend(route)
            probability[used_nodes] = -100
            probability = softmax(probability, axis=0)
            if len(route) >= self.min_num_stops_per_route:
                self.routes.append(route)

    def check_connectivity(self) -> None:
        """
        checks which node of route_x is connectes to which node route_y
        """
        self.r2r_transfer_nodes = {}
        self.r2r_connectivity = {route_id: 0 for route_id in range(len(self.routes))}
        for i in range(len(self.routes)):
            for j in range(len(self.routes)):
                if i != j:
                    node_connectivity = np.isin(self.routes[i], self.routes[j])
                    u, v = max(i, j), min(i, j)
                    self.r2r_transfer_nodes[(u, v)] = node_connectivity
                    self.r2r_connectivity[i] += node_connectivity.any()

    def fix_zero_connectivity(self) -> None:
        """
        Connects isolated routes with a group of connected routes
        """
        zero_connectivity_routes = [
            k for k, v in self.r2r_connectivity.items() if v == 0
        ]
        for z_route_id in zero_connectivity_routes:
            loc = np.random.randint(0, len(self.routes[z_route_id]))

            route_id = np.random.randint(0, len(self.routes))
            while route_id == z_route_id and len(self.routes) > 1:
                route_id = np.random.randint(0, len(self.routes))

            v = np.random.choice(self.routes[route_id])
            self.routes[z_route_id].insert(loc, v)

    def show(
        self,
        node_color: str = "lightblue",
        node_size: int = 200,
        font_size: float = 10,
        show_label: bool | None = None,
        with_labels: bool | None = True,
        ax: object = None,
        black_edges = False
    ) -> None:
        """
        Displays the created topology using `nx.spring_layout`
        """
        if ax is None:
            ax = plt.subplot(1, 1, 1)

        unique_labels = list(
            set(data["label"] for _, _, data in self.topology.edges(data=True))
        )
        colors = plt.get_cmap("tab10", len(unique_labels))
        label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
        pos = nx.spring_layout(self.topology, seed=self.seed)

        nx.draw(
            self.topology,
            pos,
            with_labels=with_labels,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
            ax=ax,
        )

        if not black_edges:
            if show_label is None:
                for label in unique_labels:
                    edges_in_group = [
                        (u, v)
                        for u, v, data in self.topology.edges(data=True)
                        if data["label"] == label
                    ]

                    nx.draw_networkx_edges(
                        self.topology,
                        pos,
                        edgelist=edges_in_group,
                        edge_color=label_color_map[label],
                        width=2,
                        label=label,
                        ax=ax,
                    )
            else:
                label = show_label
                edges_in_group = [
                    (u, v)
                    for u, v, data in self.topology.edges(data=True)
                    if data["label"] == label
                ]

                nx.draw_networkx_edges(
                    self.topology,
                    pos,
                    edgelist=edges_in_group,
                    edge_color=label_color_map[label],
                    width=2,
                    label=label,
                    ax=ax,
                )

            handles = [
                plt.Line2D([0], [0], color=label_color_map[label], lw=2, label=label)
                for label in unique_labels
            ]
            ax.legend(handles=handles, title="Edge Labels", loc="upper left")
        # ax.set_title("Network Topology with Edge Groups Colored")

        if ax is None:
            plt.show()

    def show_report(self) -> None:
        """
        Display report for the etire topology
        """
        num_nodes = self.topology.number_of_nodes()
        num_exits = len(self.exit_nodes)
        num_transfers = len(self.transfer_nodes)

        population = [node.population for node in self.nodes]
        area = [node.catchment_area_km2 for node in self.nodes]
        transit_usres = [node.transit_users for node in self.nodes]

        total_population = int(sum(population))
        total_transit_usres = int(sum(transit_usres))
        total_area = int(sum(area))

        print(f"Number of Nodes: {num_nodes}")
        print(f"Number of Exits: {num_exits}")
        print(f"Number of Transfer Nodes: {num_transfers}")
        print(f"Total Population: {total_population} persons")
        print(f"Total Transit Users: {total_transit_usres} persons")
        print(f"Total Area: {total_area} km2")

        ax = plt.subplot(1, 1, 1)
        pd.DataFrame(
            {"Population": population, "Area": area, "Transit Users": transit_usres}
        ).plot(kind="bar", figsize=(30, 6), ax=ax)
        plt.yscale("log")
        plt.show()

        plt.figure(figsize=(20, 25))
        od_mat_i = np.stack(
            [
                self.get_od_mat_for_time(i)
                for i in range(0, self.hours_of_opperation_per_day * 3600, self.analysis_period_sec)
            ]
        )
        slices = [slice(i, j) for i, j in zip(range(0, self.hours_of_opperation_per_day*60, 5), range(5, self.hours_of_opperation_per_day*60+1, 5))]

        num_pov = 8
        pov = np.concatenate(
            [
                np.random.choice(self.schools, 2),
                np.random.choice(self.offices, 2),
                np.random.choice(self.shopping, 2),
                np.random.choice(self.residentials, 2),
            ]
        )

        departures = [[od_mat_i[s, i, :].sum() for s in slices] for i in pov]
        arrivals = [[od_mat_i[s, :, i].sum() for s in slices] for i in pov]

        j = 1
        for i in range(num_pov):
            plt.subplot(num_pov, 2, j)
            plt.plot(departures[i], label=f"Departures from {pov[i]}")
            plt.legend()
            j += 1
            plt.subplot(num_pov, 2, j)
            plt.plot(arrivals[i], label=f"Arrivals to {pov[i]}")
            j += 1
            plt.legend()
        plt.show()
