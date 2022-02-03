FROM public.ecr.aws/docker/library/python:3.8

ARG TWINE_USERNAME
ARG TWINE_PASSWORD

ENV TWINE_USERNAME=$TWINE_USERNAME
ENV TWINE_PASSWORD=$TWINE_PASSWORD
ENV TWINE_NON_INTERACTIVE=true

WORKDIR /ntient

# install dependencies
COPY . .

RUN python setup.py test
RUN python setup.py sdist
RUN twine upload dist/*
