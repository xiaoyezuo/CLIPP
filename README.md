## VLA-Nav
Embedding Vision-Language-Actions for learning robot navigation

## Running Docker
Build the image with `docker build --rm -t vla-docker .`. You'll have to change the username at the bottom of `Dockerfile`. Once it is built run with `./docker_run.sh`. Once again you'll need to file path after `--volume` to where your data is. You can also configure what gpus you are using, use `--gpus all` to use all of them. Once in the image activate the conda environement with `conda activate habitat`.
