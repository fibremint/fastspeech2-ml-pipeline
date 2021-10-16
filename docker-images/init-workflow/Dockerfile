FROM python:3.8-alpine

RUN mkdir /workspace
COPY fastspeech2_env.json /workspace
COPY ./configs /workspace/configs

WORKDIR /workspace

# ENTRYPOINT [ "python", "app/entrypoint.py" ]
CMD [ "sh" ]