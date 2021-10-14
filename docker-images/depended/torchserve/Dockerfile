# install torchserve from scratch. 
# officially served torchserve image at this time uses python==3.6
# and this causes python package version mismatch, due to packages in 
# package requirement.txt requires python==3.8
# FROM python:3.8-buster
# FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
FROM fibremint/fs2-runtime-base:latest

# # ref: https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
# ENV TZ=Asia/Seoul
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install package dependencies
RUN apt-get update && \
    # apt-get install -y sudo libsndfile-dev python3.8 python3-pip python3.8-dev build-essential git && \
    git clone https://github.com/pytorch/serve pytorch-serve && \
    cd pytorch-serve && \
    python3 ./ts_scripts/install_dependencies.py --cuda=cu111 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $HOME/workspace
WORKDIR $HOME/workspace

COPY ./requirements.txt /workspace
RUN pip install -r requirements.txt
RUN pip cache purge
# download files to prevent download these in every running session
# ref: https://stackoverflow.com/questions/8590370/what-is-nltk-pos-tagger-asking-me-to-download
# RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger')"
# RUN python3 -c "from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN; InterfaceHifiGAN(model_name='hifi_gan_v1_universal')"

ENV LOG_LOCATION /var/log

COPY ./model-server /home/model-server/
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp

COPY ./dockerd-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
ENTRYPOINT [ "/usr/local/bin/dockerd-entrypoint.sh" ]

CMD ["serve"]
