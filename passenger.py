from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node import Node

import numpy as np


class Passenger:
    """
    This is a class to represent a passenger.
    A passenger has many attributes and some of them are
    `origin`, `destination`, `waiting time`, `travel time`
    """

    def __init__(
        self,
        origin: Node,
        destination: Node,
        queued_since: int,
        transfers: list[Node],
        path: list[Node],
    ) -> None:
        """
        Argument:
        --------
        `origin`: is the `Node` from where the passenger generated
        `destination`: is the final `Node` to where the passenger is wiling to go
        `queued_since`: is the time of the arrival of passenger at origin
        `transfers`: is the list of all the transfers required to go from `origin` to `destination`
        `path`: is the list of station the passenger will go through.
        """
        self.ID = np.random.randint(0, 1E9)
        self.origin = origin
        self.destination = destination
        self.queued_since = queued_since
        self.started_at = queued_since
        self.transfers = transfers

        self.waiting_time = 0
        self.stranding_counts = 0
        self.travel_time = 0

        self.path = path
        self.distance_traversed: float = 0.0
        self.num_stations_traversed: int = len(path)
        self.average_travel_speed: float = 0.0
        self.total_time_taken: float = 0.0
        self.tagged_waiting_time: int = 0


    def step(self, node:Node):
        min_exit_node = min(node.exit_nodes)
        path = [node.node_id for node in self.path]
        try:
            exit_index = path.index(min_exit_node)
        except:
            npd = {n:n.affliated_route_ids for n in self.path}
            last_route_node = [n for n in npd if list(npd[n])[0] in node.affliated_route_ids][-1]
            if last_route_node in node.od_route[min_exit_node]:
                self.is_reversed = False
                return
            else:
                self.is_reversed = True
                return
        
        node_index = path.index(node.node_id)
        if exit_index == node_index:
            self.is_reversed = True
            return

        path = path[min(exit_index, node_index) : max(exit_index, node_index)] + [min_exit_node]
        for node in [self.destination, *self.transfers]:
            if node.node_id in path:
                self.is_reversed = False
                return
            
        self.is_reversed = True
                
    def to_dct(self) -> dict:
        """
        Produce a dictionary from the passenger's data
        """
        return {
            "origin": self.origin.node_id,
            "destination": self.destination.node_id,
            "num_transfers": len(self.transfers),
            "transfers": ",".join([str(i.node_id) for i in self.transfers]),
            "waiting_time": self.waiting_time,
            "travel_time": self.travel_time,
            "stranding_counts": self.stranding_counts,
            "distance_traversed": self.distance_traversed,
            "num_stations_traversed": self.num_stations_traversed,
            "average_travel_speed": self.average_travel_speed,
            "total_time_taken": self.total_time_taken,
        }
