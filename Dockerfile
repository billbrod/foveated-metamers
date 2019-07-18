FROM continuumio/miniconda3
ARG conda_env=metamers

# we need to get gcc
RUN apt -y update
RUN apt -y install software-properties-common
RUN add-apt-repository -y "deb http://us.archive.ubuntu.com/ubuntu/ bionic main"
RUN add-apt-repository -y "deb http://us.archive.ubuntu.com/ubuntu/ bionic universe"
RUN apt -y install linux-headers-amd64 gcc

# we'll also need ffmpeg
RUN apt -y install ffmpeg

# set up our conda environment (since we grabbed the miniconda docker
# image, we already have conda installed)
ADD foveated-metamers/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN echo "source activate $conda_env" > ~/.bashrc
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH

# activate our conda environment
RUN /bin/bash -c "source activate $conda_env"

# for now, need to copy this over manually
COPY plenoptic /src/plenoptic
RUN pip install -e /src/plenoptic/

# copy over this directory
COPY foveated-metamers /src/foveated-metamers
WORKDIR /src/foveated-metamers

# Correct the config file
RUN mkdir -p /data/metamers
RUN echo "DATA_DIR: \"/data/metamers\"" > /src/foveated-metamers/config.yml

# download the seed images and place them in the correct location
RUN wget -qO- https://osf.io/5t4ju/download | tar xvz -C /data/metamers/

CMD snakemake -prk /data/metamers/seed_images/nuts_constant.pgm
