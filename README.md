
# Fashion Mnist App with Kafka
# change 1
This repository is basically a Message Broker Service using Apache Kafka for Image Recognition Model.


## Contents

- Setup and Installation
- Requirements
- How to use


## Setup and Installation

To use this project, clone the repo using the following command:

```git clone https://github.com/makaveli10/fashion-mnist-kafka-app.git```


## Requirements

- In order to use aerospike db download, setup and install aerospike from [here](https://developer.aerospike.com/docs/getting-started/linux/install-on-ubuntu)
- In order to use the Kafka, you are required to setup and start Apache Kafka Zookeeper and Kafka Server. Follow [this](https://kafka.apache.org/quickstart) guide.


## How to use

- Install the requirements:

```sh
python3 -m pip install requirements.txt
```

- To train the model, use:

```sh
python3 train.py
```

- To run the application server with ImageStream Consumer and Result Producer use:

```sh
python3 app.py
```

- [Only for Testing] This will basically stream test set images to a topic which will be consumed by the application server ImageStream Consumer. After consuming the images streams server will try to publish results under a topic and Result consumer on the client side will try and consume those results. To run client side ImageStream Producer and Result Consumer run:

```sh
python3 api.py
```


