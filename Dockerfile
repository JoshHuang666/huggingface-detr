FROM huggingface/transformers-pytorch-gpu

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=arg \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \ 
    locales \
    cmake \
    unzip \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

##################################### PIP ######################################

RUN pip3 install \
    matplotlib \
    pandas \
    quart \
    pytest-skip-slow \
    rich \
    opencv-python
    
##################################### HuggingFace #####################################

RUN pip3 install \
    datasets \
    evaluate \
    transformers[sentencepiece] \
    accelerate \
    timm \
    albumentations \
    pycocotools

##################################### Copy HuggingFace-DETR Files #####################################

COPY ./ ${HOME}/huggingface-detr

##################################### setting ###################################################

RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}/huggingface-detr

CMD ["/bin/bash"]