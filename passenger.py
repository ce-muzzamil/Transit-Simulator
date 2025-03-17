from __init__ import *

class Passenger:
    """
    This is a class to represent a passenger. 
    A passenger has many attributes and some of them are
    `origin`, `destination`, `waiting time`, `travel time` 
    """
    def __init__(self, 
                 origin: Node, 
                 destination: Node, 
                 queued_since: int,
                 transfers: list[Node],
                 ) -> None:
        """
        Argument:
        --------
        `origin`: is the `Node` from where the passenger generated
        `destination`: is the final `Node` to where the passenger is wiling to go
        `queued_since`: is the time of the arrival of passenger at origin
        `transfers`: is the list of all the transfers required to go from `origin` to `destination`
        """
        self.origin = origin
        self.destination = destination
        self.queued_since = queued_since

        self.transfers = transfers
    
        self.waiting_time = 0
        self.stranding_time = 0
        self.travel_time = 0

        self.is_aboard = False
        self.is_dropped = False
