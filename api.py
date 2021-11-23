import numpy as np
import json
import pandas as pd
from client_api.consumer import ResultConsumer

from config import get_logger
from config import Cfg
from client_api import ImageStreamProducer


def get_images_and_labels(test_df_path):
    test_df = pd.read_csv(test_df_path)
    test_df = list(test_df.values)
    label = []
    image = []
        
    for i in test_df:
        label.append(i[0])
        image.append(i[1:])

    labels = np.asarray(label)
    images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')
    return images, labels


if __name__ == "__main__":
    logger = get_logger(logger_name = __name__)
    logger.info("Api Initialized...")

    cfg = Cfg()
    result_consumer_cfg = cfg.kafka_cfg

    imgStreamProducer = ImageStreamProducer(logger)
    resultConsumer = ResultConsumer('test-topic-send', logger, result_consumer_cfg)

    # read images and stream them to topic
    images, labels = get_images_and_labels(cfg.application_cfg['test_df_path'])
    print(len(labels), images.shape)

    # stream images 
    for i, (l, im) in enumerate(zip(labels, images)):
        # stream bytes
        pushOut = imgStreamProducer.stream_records([i], [im.tolist()])
        if i==400:
            break

    # consume resutls
    results = dict()
    while True:
        consumedMessage = resultConsumer.consume_images(bufferSize=200)
        for i in range(len(consumedMessage)):
            x = json.loads(consumedMessage[i])
            for k, v in x.items():  
                results[k] = v
        with open("results.json", "w") as outfile:
            json.dump(results, outfile)