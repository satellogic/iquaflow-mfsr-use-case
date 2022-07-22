PROJ_NAME=iquaflow
CONTAINER_NAME="${PROJ_NAME}-${USER}"

ifndef DS_VOLUME
	DS_VOLUME=/scratch
endif

ifndef NB_PORT
	NB_PORT=8888
endif

ifndef MLF_PORT
	MLF_PORT=5000
endif

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"
	@echo "notebookshell -- launches a notebook server"
	@echo "mlflow -- launches an mlflow server"

build:
	docker build -t $(PROJ_NAME) .
	chmod 775 ./download.sh
	./download.sh

dockershell:
	docker run --rm --name $(CONTAINER_NAME) --gpus all -p 9197:9197 \
	-v $(shell pwd):/iqf -v $(DS_VOLUME):/data \
	-it $(PROJ_NAME)

notebookshell:
	docker run --name $(CONTAINER_NAME)-nb --gpus all --privileged -itd --rm \
	-p ${NB_PORT}:${NB_PORT} \
	-v $(shell pwd):/iqf \
	-v $(DS_VOLUME):/data \
	$(PROJ_NAME) \
	jupyter notebook \
	--NotebookApp.token='IQF' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=${NB_PORT}

mlflow:
	docker run --name $(CONTAINER_NAME)-mlf --privileged -itd --rm \
	-p ${MLF_PORT}:${MLF_PORT} \
	-v $(shell pwd):/iqf -v $(DS_VOLUME):/data \
	$(PROJ_NAME) \
	mlflow ui --host 0.0.0.0:$(MLF_PORT)
