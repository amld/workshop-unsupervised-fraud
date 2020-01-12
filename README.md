# amld2020-unsupervised
Repository for the Fraud Detection (outlier detection on mixed data) workshop, AMLD 2020  

## Cloning the repo
To clone the repo:

`mkdir <DIR>`

`cd <DIR>`

`git clone https://github.com/DonErnesto/amld2020-unsupervised`

`cd amld2020-unsupervised/`



## Docker instructions
The notebooks in the folder /notebooks depend on scikit-learn and pyod, which in turn requires Keras and Tensorflow for certain models. It is advised to use the Docker image provided in docker-python, as it comes with all necessary packages.

This image is based on the docker image `jupyter/tensorflow-notebook`,
see also https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-tensorflow-notebook


Installing Docker and downloading the tensorflow-notebook image requires roughly 6 GB of disk space.


To get Docker up and running,
- Download and install Docker Desktop https://www.docker.com/products/docker-desktop
- In the base directory, execute:

    `$ docker build docker-python -t jupyter-outlieramld`
    This will fetch the base image and additionally install keras and pyod
- Check that the image is built:
    `$ docker images`. The image `jupyter-outlieramld` should be there
- The following command will run the image, broadcast the notebook server to port 8888, and attach a volume (a connection to the host' filesystem)
    `$ docker run -it -p 8888:8888 -v $(pwd):/home/jovyan jupyter-outlieramld`

Copy-paste the link (`http://127.0.0.1:8888/?token=124a64...`) into a browser.

**Docker tips**
- The container can be stopped by ctrl-c in the terminal when the notebook is running (the normal way)
- `$ docker ps -a` shows all Docker containers, running and stopped
- A terminal may be opened in a running Docker container, with `$ docker exec -it <container id> bash`
- To kill all stopped Docker containers (which may save some space, no need to do so when in doubt):
`$ docker containers prune`

## Getting the data

....
