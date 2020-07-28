FROM ubuntu:18.04

RUN mkdir $HOME/preprocess
COPY requirements.txt $HOME/preprocess/requirements.txt
RUN $HOME/preprocess/requirements.txt

RUN mkdir $HOME/preprocess/data

COPY preprocess.py $HOME/preprocess
