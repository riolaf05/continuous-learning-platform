FROM ubuntu:18.04

RUN mkdir /preprocess
COPY requirements.txt /train/requirements.txt

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r /train/requirements.txt

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN mkdir /train/data

COPY train.py /train/train.py
