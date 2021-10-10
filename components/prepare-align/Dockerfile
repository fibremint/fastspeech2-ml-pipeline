FROM python:3.8-buster

# install dependency
RUN apt-get update && \
    apt-get install -y libsndfile-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace
WORKDIR /workspace

# install package dependencies
COPY ./requirements.txt /workspace
RUN pip install -r requirements.txt

COPY ./app /workspace/app

ENTRYPOINT [ "python", "app/entrypoint.py" ]
