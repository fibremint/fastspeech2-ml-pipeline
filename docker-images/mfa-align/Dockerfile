FROM ubuntu:20.04

# ref: https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install dependencies
RUN apt-get update && \
    apt-get install -y libopenblas-base libsndfile-dev wget && \
    rm -rf /var/lib/apt/lists/*

# install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# install MFA
RUN conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch
SHELL ["conda", "run", "-n", "aligner", "/bin/bash", "-c"]
RUN pip install montreal-forced-aligner
RUN mfa thirdparty download
RUN mfa download acoustic english

# run MFA alignment
# COPY ./entrypoint.py /
# ref: https://pythonspeed.com/articles/activate-conda-dockerfile/
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "aligner", "python", "/entrypoint.py"]
