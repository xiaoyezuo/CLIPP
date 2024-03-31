FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as runtime
#FROM nvidia/cudagl:11.1.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# system depends
RUN apt-get install -y --no-install-recommends gcc cmake
RUN apt-get install -y --no-install-recommends python3-pip git python3-dev python3-opencv libglib2.0.0 wget python3-pybind11

#RUN python3 -m pip intsall --upgrade pip

# torch install
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip3 install jupyterlab notebook


#install CLIP-ViL dependencies
RUN pip3 install tqdm stanza tensorboardX

#build and install the Matterport3D sim
RUN apt-get install -y --no-install-recommends libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev

RUN git clone --recursive https://github.com/jhughes50/CLIP-ViL  && cd CLIP-ViL/CLIP-ViL-VLN && mkdir build && cd build && cmake -DOSMESA_RENDERING=ON .. && make

EXPOSE 8381

ARG GID=1000
ARG UID=1000
RUN addgroup --gid $GID jason 
RUN useradd --system --create-home --shell /bin/bash --groups sudo -p "$(openssl passwd -1 jason)" --uid $UID --gid $GID jason
USER jason
WORKDIR /home/vla-docker/

#RUN cd VLA-Nav/CLIP-ViL/CLIP-ViL-VLN && mkdir build && cd build
#RUN cmake -DEGL_RENDERING=ON .. && make -j8
