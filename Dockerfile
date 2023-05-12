# This dockerfile contains the packages needed to execute simba_ml.

FROM python:3.10.8-buster

# Install system dependencies
RUN apt-get update && apt-get install -y

# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
COPY dev_requirements.txt /tmp/dev_requirements.txt
COPY docs_requirements.txt /tmp/docs_requirements.txt
RUN pip install --upgrade pip
RUN pip install tensorflow==2.11.0
RUN pip install -r /tmp/requirements.txt
RUN pip install -r /tmp/dev_requirements.txt
RUN pip install -r /tmp/docs_requirements.txt
RUN rm /tmp/requirements.txt
RUN rm /tmp/dev_requirements.txt
RUN rm /tmp/docs_requirements.txt