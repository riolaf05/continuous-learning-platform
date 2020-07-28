FROM ubuntu:18.04

RUN mkdir /preprocess
COPY requirements.txt /preprocess/requirements.txt

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r /preprocess/requirements.txt

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN mkdir /preprocess/data

COPY preprocess.py /preprocess/preprocess.py
