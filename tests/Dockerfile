FROM python:3.7

# project/code-framework arguments (to be changed)
ARG APP=PyMLFrameWork

# define arguments (global variables) in Dockerfile
ARG DEST=/code
ARG APP_PORT=8080
ARG MLFLOW_PORT=5000
ARG DEBIAN_FRONTEND=noninteractive

# define environmental variables Docker Container
ENV DEBUG="false"
ENV MLFLOW_SERVER="remote"
ENV APP_NAME=$APP
ENV APP_LOCATION=$DEST
ENV PYTHON_PATH=/usr/local/bin
ENV DEST=$DEST
ENV APP_PORT=$APP_PORT
ENV MLFLOW_PORT=$MLFLOW_PORT
ENV SSH_PASSWD "root:Docker!${APP}!"
ENV TZ="Asia/Kuala Lumpur"

# update and install new linux packages
RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
	&& echo "$SSH_PASSWD" | chpasswd \
        && apt-get install -y --no-install-recommends vim \
        && apt-get install -y --no-install-recommends supervisor \
        && echo "${TZ}" > /etc/timezone \
        && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get install build-essential -y  --no-install-recommends
RUN apt-get install  -y --no-install-recommends libboost-all-dev
RUN apt-get install  -y --no-install-recommends unixodbc-dev

# making new directory in docker container and set current working directory
RUN mkdir -p /var/run/sshd /var/log/supervisor
RUN mkdir $DEST
WORKDIR $DEST

# add python module requirements and install the modules
ADD src/requirements.txt .
RUN pip install gunicorn pybind11 gevent
RUN pip install -r requirements.txt

# check current python path
RUN which python

# add files from current VM to container
ADD . .

# temporary fix for mlflow integration (inside conf/temp)
COPY conf/temp/registry.py /usr/local/lib/python3.7/site-packages/mlflow/tracking/_tracking_service/
COPY conf/temp/utils.py /usr/local/lib/python3.7/site-packages/mlflow/tracking/_model_registry/

# copy supervisor config file to Docker Container
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE $APP_PORT $MLFLOW_PORT

# run supervisord
CMD ["/usr/bin/supervisord"]
