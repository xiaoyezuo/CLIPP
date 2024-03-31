#!/bin/sh
docker run --rm -it -e "TERM=xterm-256color" --gpus '"device=0,1"' -v "/raid0/docker-raid/jasonah:/home/vla-docker/data" -v ".:/home/vla-docker" vla-docker:latest
