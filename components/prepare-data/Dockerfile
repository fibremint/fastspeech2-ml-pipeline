FROM python:3.8-alpine

RUN mkdir /workspace
COPY ./app /workspace/app

WORKDIR /workspace

ENTRYPOINT [ "python", "app/entrypoint.py" ]
