import json
from kafka import KafkaProducer

class ResultProducer():
    '''
    Producer class to stream results.
    '''
    def __init__(self, logger):
        '''
        Initialzes Result Producer class.
        '''
        self._logger = logger
        self._producer = self._init_producer()

    def _init_producer(self):
        '''
        Initializes Kafka Producer.
        '''
        return KafkaProducer(value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    def stream_records(self, token_id, class_names):
        '''
        Pushes and Streams results.
        '''
        if len(token_id) == 0:
            return False
        try:
            for tid, classLabel in zip(token_id, class_names):
                self._producer.send('test-topic-send', {tid: classLabel})
            self._producer.flush()
            self._logger.info(f"producer streaming records success!")
            return True
        except Exception as e:
            self._logger.error(f"Failed to stream Records!{e}")
            return False
