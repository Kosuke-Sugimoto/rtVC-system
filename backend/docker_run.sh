#!/bin/bash

DOCKER_IMAGE_NAME="rtvc-be:latest"
DOCKER_CONTAINER_NAME="rtvc-be"

if [[ "$(docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == "" ]]
then
    docker build -t $DOCKER_IMAGE_NAME .
fi

docker run -it \
           --name $DOCKER_CONTAINER_NAME \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --mount type=bind,src=./,dst=/work/ \
           -p 8765:8765 \
           $DOCKER_IMAGE_NAME
