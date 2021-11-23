# from kafka import KafkaProducer


# producer = KafkaProducer(bootstrap_servers='localhost:9092')

# producer.send('vector-test', s.encode('utf-8'))
# producer.flush()

import json
import pandas as pd
from kafka import KafkaProducer
from config import get_logger


class ImageStreamProducer():
    '''
    Producer class to stream images.
    '''
    def __init__(self, logger):
        '''
        Initialzes Image stream Producer class.
        '''
        self._logger = logger
        self._producer = self._init_producer()

    def _init_producer(self):
        '''
        Initializes Kafka Producer.
        '''
        return KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    def stream_records(self, token_id, image_stream):
        '''
        Pushes and Streams images.
        '''
        if len(token_id) == 0:
            return False
        try:
            for tid, img in zip(token_id, image_stream):
                self._producer.send('image-stream-topic', {tid: img})
            self._producer.flush()
            self._logger.info(f"producer streaming records success!")
            return True
        except Exception as e:
            self._logger.error(f"Failed to stream Records!{e}")
            return False    

