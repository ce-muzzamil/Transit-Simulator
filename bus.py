import numpy as np
from topology import Topology
from node import Node
from passenger import Passenger
import networkx as nx


class Bus:
    """
    This class simulates a single bus in a transit system. It oppertares on a predefined subset of topology i.e., `route_id`.
    if the route of `route_id` comprises if N number of stations then it wil start jouney from point S1 and end on S10.
    After this completion the bus will restart journey from S9 to S0. where S is the Sn is any station in the route.
    """

    def __init__(
        self,
        capacity: int,
        avg_speed: float,
        service_route: int,
        step_interval: int,
        topology: Topology,
        reversed: bool,
        analysis_period_sec: int = 60,
    ):
        """
        Arguments:
        ---------
        `capacity`: is the number of passengers that the bus can carry
        `avg_speed: is the average speed the bus is going to have during motion`
        `service_route`: is the id of the predifined subsets of the `topology`
        `topology`: is the instance of `Topology` class that represents the transit system
        `reversed`: is the boolen defining the direction of bus. i.e., A->B or B->A
        `analysis_period_sec`: is the time interval in seconds for which the bus is going to be analyzed
        """
        self.ID = np.random.randint(0, 1e9)
        self.capacity = capacity
        self.avg_speed = avg_speed
        self.speed = avg_speed  # kmph
        self.service_route = service_route
        self.location = 0.0
        self.step_interval = step_interval
        self.reversed = reversed
        self.total_distance_traversed = 0
        self.num_passengers_served = 0
        self.analysis_period_sec = analysis_period_sec

        self.topology = topology
        nodes_ids = [
            [route.node_u.node_id, route.node_v.node_id]
            for route in self.topology.routes
            if route.route_id == self.service_route
        ]

        nodes_ids = list(set(sum(nodes_ids, [])))
        self.num_stations_in_trajectory = len(nodes_ids)

        subgraph: nx.Graph = self.topology.topology.subgraph(nodes_ids)
        subgraph = nx.Graph(subgraph)

        to_drop = []
        for u, v, data in subgraph.edges(data=True):
            if data["label"] != service_route:
                to_drop.append((u, v))
                to_drop.append((v, u))

        subgraph.remove_edges_from(to_drop)

        self.neighbors = {
            node: list(nx.neighbors(subgraph, node)) for node in nodes_ids
        }

        self.exit_nodes = [
            node_id for node_id in nodes_ids if len(self.neighbors[node_id]) == 1
        ]

        self.routes = [
            route
            for route in self.topology.routes
            if route.route_id == self.service_route
        ]

        self.passengers: list[Passenger] = []
        self.initilize_trajectory()
        self.done = False

    def initilize_trajectory(self):
        """
        This methods is called every time the bus' trajectory needs initilization.
        The methods initlizes the tragectory `self.to_go` based on the `reversed` flag.
        The trajectory is the itinerary (an ordered list) for the bus that includes all the nodes the bus is going to visit.
        Based on this list and the `Node` data and the traectory, this method calculates the distances between all the `to_go` nodes.
        """
        try:
            if self.reversed:
                self.routes = self.routes[::-1]
                current_node_id = min(self.exit_nodes)
            else:
                current_node_id = max(self.exit_nodes)
        except:
            raise Exception("Error in Bus initiallization", 
                            f"{self.service_route=}, {self.topology.seed=}")

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

        self.distances = [0]
        for node_pair in zip(self.to_go[:-1], self.to_go[1:]):
            for route in self.routes:
                if all([node in route.node_pair for node in node_pair]):
                    self.distances.append(route.distance)
                    break

        self.distance_next_node = self.distances.pop(0)

    def get_node_by_id(self, id: int) -> Node:
        """
        To get the instance of `Node` given the `node_id` in the topology

        Argument:
        --------
        id: is the node id of the transit station: `Node`

        Return:
        ------
        Transit Station: `Node`
        """
        for node in self.topology.nodes:
            if node.node_id == id:
                return node

    def step(self, time: int) -> list[Passenger]:
        """
        A method that will be repeatedly called to perform opperations like.
        1- To record instantaneous location
        2- To record instantaneous speed
        3- To call the `step` method of the `Node` it crosses
        4- To change the state of `reversed` flag if the last stops is reached

        Argument:
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation

        Returns:
        list of passengers that have reached destination
        """

        to_drop = []
        if self.distance_next_node <= 0:
            self.current_node = self.to_go.pop(0)
            to_drop = self.current_node.bus_arrived(time, self)
            if len(self.distances) > 0:
                distance_next_node = self.distances.pop(0)
                self.distance_next_node = distance_next_node - abs(
                    self.distance_next_node
                )

        if len(self.to_go) == 0:
            self.done = True
            return to_drop

        self.speed = min(
            max(5.56, self.avg_speed + np.random.normal(loc=0, scale=10)), 33.34
        )
        self.distance_next_node -= self.speed * self.step_interval
        self.total_distance_traversed += self.speed * self.step_interval
        self.num_passengers_served += len(to_drop)

        for passenger in self.passengers:
            passenger.travel_time += self.analysis_period_sec

        return to_drop
