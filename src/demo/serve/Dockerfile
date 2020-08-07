FROM ubuntu:18.04

RUN mkdir /serve
COPY requirements.txt /serve/requirements.txt

RUN apt-get update && apt-get install -y python3-pip curl
RUN pip3 install -r /serve/requirements.txt
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN sha256sum Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b

RUN apt-get install -y libsm6 libxext6 libxrender-dev


