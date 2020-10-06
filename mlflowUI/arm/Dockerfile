FROM raspbian/stretch

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --no-cache-dir Cython

RUN python3 -c "import Cython; print(Cython.__version__)"

RUN pip3 install mlflow

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

CMD ["mlflow", "server", "--backend-store-uri", "/tmp/mlflowruns", "--host", "0.0.0.0"]
