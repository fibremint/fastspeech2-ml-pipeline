FROM python:3.8-alpine

RUN mkdir /workspace
# COPY fastspeech2_env.json /workspace
COPY ./default-data /workspace/default-data

WORKDIR /workspace

# ENTRYPOINT [ "python", "app/entrypoint.py" ]
CMD [ "sh" ]