FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# as runtime
#FROM nvidia/cudagl:11.1.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# system depends
RUN apt-get install -y --no-install-recommends gcc cmake sudo
RUN apt-get install -y --no-install-recommends python3-pip git python3-dev python3-opencv libglib2.0.0 wget python3-pybind11 vim

#RUN python3 -m pip intsall --upgrade pip

# torch install
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip3 install jupyterlab notebook

# install conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# habitat conda install
RUN conda create -n habitat python=3.9 cmake=3.14.0 && conda init bash && conda activate habitat && conda install habitat-sim headless -c conda-forge -c aihabitat

#install CLIP-ViL dependencies
RUN pip3 install tqdm stanza tensorboardX openai-clip

#build and install the Matterport3D sim
RUN apt-get install -y --no-install-recommends libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev

#RUN mkdir -p data

ARG GID=1000
ARG UID=1000
env USER jasonah
RUN addgroup --gid $GID jasonah 
RUN useradd --system --create-home --shell /bin/bash --groups sudo -p "$(openssl passwd -1 jasonah)" --uid $UID --gid $GID jasonah
WORKDIR /home/vla-docker/

USER $USER
WORKDIR /home/vla-docker
CMD ["bash"]
