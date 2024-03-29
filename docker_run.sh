#!/bin/sh
docker run --rm -it -e "TERM=xterm-256color" --gpus '"device=0,1"' --volume="/raid0/docker-raid/jasonah:/home/vla-docker"  vla-docker:latest
