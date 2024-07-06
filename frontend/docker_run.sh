#!/bin/bash

DOCKER_IMAGE_NAME="rtvc-fe:latest"
DOCKER_CONTAINER_NAME="rtvc-fe"

if [[ "$(docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == "" ]]
then
    docker build -t $DOCKER_IMAGE_NAME .
fi

docker run -it \
           --name $DOCKER_CONTAINER_NAME \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --mount type=bind,src=./,dst=/work/ \
           -p 5173:5173 \
           $DOCKER_IMAGE_NAME
