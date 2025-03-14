from __init__ import *

class Passenger:
    def __init__(self, origin, 
                 destination, 
                 queued_since,
                 transfers,
                 ):
        self.origin = origin
        self.destination = destination
        self.queued_since = queued_since

        self.transfers = transfers
    
        self.waiting_time = 0
        self.stranding_time = 0
        self.travel_time = 0

        self.is_aboard = False
        self.is_dropped = False
        