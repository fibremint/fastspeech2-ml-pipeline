# install torchserve from scratch. 
# officially served torchserve image at this time uses python==3.6
# and this causes python package version mismatch, due to packages in 
# package requirement.txt requires python==3.8
# FROM python:3.8-buster

# RUN mkdir $HOME/workspace
# WORKDIR $HOME/workspace

# # install package dependencies
# RUN apt-get update && \
#     apt-get install -y sudo libsndfile-dev && \
#     git clone https://github.com/pytorch/serve pytorch-serve && \
#     cd pytorch-serve && \
#     python ./ts_scripts/install_dependencies.py && \
#     rm -rf /var/lib/apt/lists/*

# COPY ./requirements.txt /workspace
# RUN pip install -r requirements.txt

# RUN apt-get update && \
#     apt-get install -y libsndfile-dev && \
#     rm -rf /var/lib/apt/lists/*
FROM fibremint/fs2-torchserve

RUN mkdir $HOME/workspace
WORKDIR $HOME/workspace

COPY ./app $HOME/workspace/app

ENTRYPOINT [ "python", "app/entrypoint.py" ]
