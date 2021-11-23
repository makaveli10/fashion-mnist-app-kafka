from kafka import KafkaConsumer

class ResultConsumer():
    def __init__(self, topics, logger, cfg):
        self._topics = topics
        self._cfg = cfg
        self._logger = logger
        self._consumer = self._create_consumer()

    def _create_consumer(self):
        return KafkaConsumer(
            self._topics,
            bootstrap_servers=self._cfg["bootstrap_servers"],
            auto_offset_reset=self._cfg["auto_offset_reset"], 
            client_id=self._cfg["client_id"],
            group_id=self._cfg["group_id"],
            enable_auto_commit=True
        )

    def consume_images(self, bufferSize=1):
        fetched_msg, consumed = [], 0
        for message in self._consumer:
            try:
                fetched_msg.append(message.value.decode("utf-8"))
                consumed+=1
                if consumed > bufferSize:
                    break
            except Exception as e:
                self._logger.warning(f"Failed to Fetch messages{e}!")

        return fetched_msg    