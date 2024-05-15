ARG CUDA_VERSION=11.1-cudnn8
ARG OS_VERSION=ubuntu20.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${OS_VERSION} as git

RUN apt-get update && apt-get install -y openssh-client git

WORKDIR /opt/
RUN git clone https://github.com/vicoslab/CeDiRNet.git

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${OS_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="domen.tabernik@fri.uni-lj.si"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-opencv

WORKDIR /opt/
COPY --from=git /opt/CeDiRNet/src  /opt/CeDiRNet
COPY --from=git /opt/CeDiRNet/requirements.txt  /opt/requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "/opt/CeDiRNet/infer.py"]

