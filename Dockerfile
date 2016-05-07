FROM ubuntu:latest
RUN apt-get update && apt-get -y install python-dev build-essential git x11-apps
RUN apt-get install -y python-setuptools python-pip python-pygame python-matplotlib python-numpy python-scipy
RUN pip install cv2
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
RUN git clone https://github.com/Theano/Theano.git
RUN cd ./Theano && python setup.py develop
COPY ./ /opt/PyDataLondon2016
ENV PYTHONPATH /opt/PyDataLondon2016/
WORKDIR /opt/PyDataLondon2016
RUN git submodule init && git submodule update
