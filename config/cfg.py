import os
import pathlib

from jproperties import Properties
from config.constants import *
from config.utils import read_json


# Defining Parent Path
path = pathlib.Path(__file__).resolve().parent.parent
print(path)

class Cfg:
    '''Configuration class to load all the configs at once.
    '''
    def __init__(self) -> None:
        self.kafka_cfg = self.get_kafka_config()
        self.aerospike_cfg = self.get_aero_spike_config()
        self.application_cfg = self.get_app_config()

    def get_kafka_config(self):
        '''
        Reads kafka configuration files and loads
        consumer and producer
        Returns:
            Dictionary with producer and consumer configurations.
        '''
        return read_json(os.path.join(path, consumer_cfg_path))

    def get_aero_spike_config(self):
        '''
        Reads aerospike configuration file. Load 
        Configuration for Aerospike.
        Returns:
            Aerospike Configuration.
        '''
        return read_json(os.path.join(aerospike_cfg_path))


    def get_app_config(self):
        '''
        Reads app configuration file and loads app config.
        Returns:
            Application Configuration.
        '''
        return read_json(os.path.join(app_cfg_path))