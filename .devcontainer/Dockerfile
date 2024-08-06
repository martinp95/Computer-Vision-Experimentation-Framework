ARG TF_VERSION="2.16.1"
FROM tensorflow/tensorflow:${TF_VERSION}-gpu


RUN apt update && apt install -y python3 python3-pip python-is-python3 git wget unzip libgl1-mesa-glx numactl\
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Create folder to store notebooks
#RUN mkdir /workspace/jupyter-dev_env/notebooks/

## Install jupyter labs and tools
RUN pip3 install jupyterlab jupyter ipywidgets ipykernel

## Install other utilities
RUN pip3 install matplotlib pandas opencv-python seaborn scikit-learn tensorflow

## Workdir folder to store the notebooks
WORKDIR /workspaces/Computer-Vision-Experimentation-Framework