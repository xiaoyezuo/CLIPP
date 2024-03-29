## VLA-Nav
Embedding Vision-Language-Actions for learning robot navigation

## Running Docker
Build the image with `docker build -t vla-docker . -f Dockerfile`. You'll have to change the username at the bottom of `Dockerfile`. Once it is built run with `./docker_run.sh`. Oone again you'll need to file path after `--volume` to where your data is. You can also configure what gpus you are using, use `--gpus all` to use all of them.
