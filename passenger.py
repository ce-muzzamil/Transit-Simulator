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
