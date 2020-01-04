# amld2020-unsupervised
Repository for the Fraud Detection (outlier detection on mixed data) workshop, AMLD 2020  


## Docker instructions
The notebooks in the folder /notebooks depend on scikit-learn and pyod, which in turn requires Keras and Tensorflow for certain models. It is advised to use the Docker image provided in docker-python, as it comes with all necessary packages.
This image is based on the docker image `jupyter/tensorflow-notebook`
See also https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-tensorflow-notebook


To get Docker to work, 
- Download and install Docker Desktop
- In the base directory, execute:
    $ docker build docker-python -t jupyter-outlieramld
    This will fetch the base image and additionally install keras and pyod
- Check that the image is built by typing:
    $ docker images 
    `jupyter-outlieramld` should be there
- The following command will run the image 
