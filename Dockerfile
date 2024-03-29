FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as runtime
#FROM nvidia/cudagl:11.1.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# system depends
RUN apt-get install -y --no-install-recommends gcc
RUN apt-get install -y --no-install-recommends python3-pip git python3-dev python3-opencv libglib2.0.0 wget

#RUN python3 -m pip intsall --upgrade pip

# torch install
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip3 install jupyterlab notebook

EXPOSE 8381

ARG GID=1000
ARG UID=1000
RUN addgroup --gid $GID jason 
RUN useradd --system --create-home --shell /bin/bash --groups sudo -p "$(openssl passwd -1 jason)" --uid $UID --gid $GID jason
USER jason
WORKDIR /home/vla-docker/
