# amld2020-unsupervised
Repository for the Fraud Detection (outlier detection on mixed data) workshop, AMLD 2020  

## General instructions:

- By cloning the repo, you will have the notebooks and data files needed during the workshop
- To have a working environment for the code, you may either:
    - Use the colab link in the `*_colab.ipynb` notebooks (NB: a google account is needed for this!)
    - Use Docker (in that case, be sure to run the Dockerfile, see below)
    - Install the packages from the `requirements.txt`

For the workshop challenge, you will be submitting to an API that is hosted on AWS. Internet access is therefore vital! (the data quantities are however rather small).


## Cloning the repo
To clone the repo:

`mkdir <DIR>`

`cd <DIR>`

`git clone https://github.com/amld/workshop-unsupervised-fraud`

`cd workshop-unsupervised-fraud/`


## Running the exercises
During the workshop, we will work on two Notebooks, `exercises_1.ipynb` (or `exercises_1_colab.ipynb`) and `challenge.ipynb` (or `challenge_colab.ipynb`).

Instructions will be given during the workshop. 

## Getting the right Python Environments

The notebooks in the directory `/notebooks depend on packages like scikit-learn and pyod, which in turn have other dependencies. To guarantee a compatible environment, there are three options.

The first option is to use the `_colab.ipynb` notebooks that have a colab link. **For colab, a Google account is necessary**.

The second one is to use the Dockerfile that is provided in `\docker-python`. Note that Docker needs to be installed, and that the Docker image is large, almost 4GB, so this needs to be done **before the workshop**.

The final option is to create a conda environment (or other virtual environment) with the packages in `requirements.txt` installed.


### Colab instructions (option 1)
Open the jupyter notebook with the colab link (`_colab.ipynb`). This link will direct you to Google's colab.


### Docker instructions (option 2)
The Docker image is based on the docker image `jupyter/tensorflow-notebook`,
see also https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-tensorflow-notebook
This image may also be useful for other workshops.

Installing Docker and downloading the tensorflow-notebook image requires roughly **6 GB of disk space**.


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

### Creating a conda- or virtual environment
Run `pip install -r requirements.txt`
