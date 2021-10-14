FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

# ref: https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y sudo libsndfile-dev python3.8 python3-pip python3.8-dev build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY ./requirements.txt /workspace
RUN pip install -r requirements.txt
RUN pip cache purge

RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger')"
RUN python3 -c "from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN; InterfaceHifiGAN(model_name='hifi_gan_v1_universal')"
