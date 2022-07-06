FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "**** Installing Python ****" && \
    add-apt-repository -y ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.6 python3.6-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm -rf /var/lib/apt/lists/* && \
    alias python3=python3.6

RUN apt-get update &&  \
    apt-get install -y \
    curl unzip wget git tar \
    ffmpeg libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx

RUN echo 'alias pip=pip3.6' >> ~/.bashrc
RUN echo 'alias pip3=pip3.6' >> ~/.bashrc

RUN ln -sf /bin/python3.6 /bin/python
RUN ln -sf /bin/python3.6 /bin/python3

RUN pip3 install pip --upgrade

WORKDIR /mfsr

RUN pip3 install git+https://github.com/satellogic/iquaflow.git

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install rasterio==1.2.6
RUN pip3 install kornia --no-deps

CMD ["/bin/bash"]
