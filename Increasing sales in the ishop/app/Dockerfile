FROM ubuntu:22.04

ARG USER=user

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/home/${USER}/.local/bin:${PATH}"

COPY . .

RUN apt-get update \
    && adduser $USER --disabled-password --gecos "" \
    && apt-get install -y python3 python3-pip --no-install-recommends
    
USER $USER

RUN pip3 install xgboost==1.6.2 scikit-learn --user \
    && pip3 install fastapi uvicorn --user \
    && rm -rf ~/.cache/pip
    
