import numpy as np
import aerospike
import pandas as pd

from api_kafka import ResultProducer, ImageStreamConsumer
from config.cfg import Cfg
from config import get_logger
from config import Cfg
from config import weights_path

from utils.infer_utils import Classfier


def setup_db(db_cfg):
    '''
    Connect to the AeroSpike Client.
    Args:
        db_cfg: Aerospike database configuration
    Returns:
        Connected Aerospike Client
    '''
    host = db_cfg['host']
    port = db_cfg['port']
    namespace = db_cfg['namespace']

    # create aerospike client
    try:
        hosts = host.split(",")
        aerospike_cfg_dict = {'hosts': list(tuple([host, int(port)]) for host in hosts)}
        aerospike_client = aerospike.client(aerospike_cfg_dict).connect()
        return aerospike_client
    except Exception as e:
        print(e)
        return None
        

def publish_results(aerospikeClient, token_id, class_names, namespace, logger):
    '''
    Push Records to ResultProducer & Aerospike Database.
    Args:
        aerospikeClient: AeroSpike Client.
        token_id: Request ID of Messages.
        class_names: Prediction Class Labels.
        namespace: Name of the Database.
    '''
    logger.info(f"Pushing Records to NoSQL Database...")
    try:
        for token_id, classLabel in zip(token_id, class_names):
            key = ('test', 'test-table-name', str(token_id))
            if aerospikeClient or not aerospikeClient.closed():
                aerospikeClient.put(key, {str(token_id) : classLabel})
            logger.info(f"Successfully published Records...")
            return True
    except Exception as e:
        logger.error(f"Failed to publish records {e}")
        return False


if __name__ == "__main__":
    logger = get_logger(logger_name = __name__)
    logger.info("App Initialized...")
    logger.info("Loading Config files...")
    cfg = Cfg()

    consumer_cfg = cfg.kafka_cfg

    logger.info("Creating Kafka Producer & Consumer Clients")

    imageStreamConsumer = ImageStreamConsumer(
        topics="image-stream-topic", 
        cfg=consumer_cfg, 
        logger=logger
    )
    producer = ResultProducer(logger=logger)

    logger.info("Fetching Model Utilities")
    classifier = Classfier(weights_path, logger=logger)

    logger.info("Creating Aerospike Client")
    aerospike_client = setup_db(cfg.aerospike_cfg)

    buffer_size = cfg.application_cfg['client.buffer.size']

    while True:
        consumed_message = imageStreamConsumer.consume_images(bufferSize=buffer_size)        

        ids, class_names = classifier.classify(consumed_message)
        pushed = producer.stream_records(
            token_id=ids, 
            class_names=class_names
        )

        pushed_aero_spike = publish_results(
            aerospike_client, 
            token_id = ids, 
            class_names=class_names, 
            namespace=cfg.aerospike_cfg['namespace'], 
            logger=logger
        )

        logger.info(f"Producer publish results Status: {pushed}, Aerospike push Status {pushed_aero_spike}")

