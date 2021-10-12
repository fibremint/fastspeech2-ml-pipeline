FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

# ref: https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y libsndfile-dev python3.8 python3-pip python3.8-dev build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace
WORKDIR /workspace

# install package dependencies
COPY ./requirements.txt /workspace
RUN pip install -r requirements.txt

COPY ./app /workspace/app

ENTRYPOINT [ "python3", "app/entrypoint.py" ]
