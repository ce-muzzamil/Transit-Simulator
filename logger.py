import os
import csv
from __init__ import *


class PassengerLogger:
    def __init__(self, path:str):
        self.path = path
        self.fields = ('''origin
                       destination
                       num_transfers
                       transfers
                       waiting_time
                       travel_time
                       ''').split('\n')
        
        self.fields = [x.strip() for x in self.fields]
        self.write()
        self.reset_pool()

    def write(self) -> None:
        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.fields)

    def reset_pool(self) -> None:
        self.pool = {k: None for k in self.fields}

    def commit(self) -> None:
        with open(self.path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.pool.values())
        self.reset_pool()

    def add_to_pool(self, **kwargs) -> None:
        for var, val in kwargs.items():
            if var in self.fields:
                self.pool[var] = val
            else:
                raise Exception('Unexpected field ', var)
            