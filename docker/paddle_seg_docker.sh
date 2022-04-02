#!/usr/bin/env bash

PERCEPTION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "paddle seg rootdir ${PERCEPTION_DIR}"
docker_name="paddle-seg-lane"
#docker_image_addr="master2:5000/paddle-seg-tjk"
docker_image_addr="master2:5000/paddle-seg-tjk"

function docker_init () {
	docker pull ${docker_image_addr}
	docker_del
	docker_run
}

function docker_run () {
	num=`docker ps -a | grep -w ${docker_name} | wc -l`
	if [ $num -eq 0 ];then
		docker run -it \
		  --runtime=nvidia \
			--privileged=true \
			--shm-size=6g \
			--net=host \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-e NVIDIA_VISIBLE_DEVICES=all \
			-e NVIDIA_DRIVER_CAPABILITIES=all \
			-e DISPLAY=unix$DISPLAY \
			-e GDK_SCALE \
			-e GDK_DPI_SCALE \
			-v /data4:/data4 \
			-v /data7:/data7 \
			-v /data8:/data8 \
			-v /data9:/data9 \
			-v /nas:/nas \
			-v /nas2:/nas2 \
			--name ${docker_name} \
			${docker_image_addr}
	elif [ $num -eq 1 ];then
		echo "docker: ${docker_name} container exist restart it"
		docker start ${docker_name}
		docker attach ${docker_name}
		echo "           "
	else
		docker ps -a | grep -w ${docker_name}
		echo "please delete no use docker container"
	fi
}

function docker_new_ternimal () {
	num=`docker ps | grep -w ${docker_name} | wc -l`
	if [ $num -eq 1 ];then
		docker exec -it ${docker_name} /bin/bash
	elif [ $num -eq 0 ];then
		echo "please init docker: ${docker_name} container first"
	else
		docker ps -a | grep -w ${docker_name}
		echo "please delete no use docker"
	fi
}

function docker_stop () {
	num=`docker ps | grep -w ${docker_name} | wc -l`
	if [ $num -eq 1 ];then
		docker stop ${docker_name}
	elif [ $num -eq 0 ];then
		echo "docker: ${docker_name} container is not start"
	else
		docker ps -a | grep -w ${docker_name}
		echo "please delete no use docker container"
	fi
}

function docker_del () {
	num=`docker ps | grep -w ${docker_name} | wc -l`
	num2=`docker ps -a | grep -w ${docker_name} | wc -l`
	if [ $num -eq 1 ];then
		docker stop ${docker_name}
		docker rm ${docker_name}
	elif [ $num -eq 0 ];then
		if [ $num2 -eq 1 ]; then
			docker rm ${docker_name}
		fi
		if [ $num2 -gt 1 ]; then
			docker ps -a | grep -w ${docker_name}
			echo "please delete no use docker container"
		fi
	else
		docker ps -a | grep -w ${docker_name}
		echo "please delete no use docker container"
	fi
}

function docker_commit() {
  num=`docker ps -a | grep -w ${docker_name} | wc -l`
	if [ $num -eq 1 ];then
	  docker commit ${docker_name} ${docker_image_addr}
	  docker push ${docker_image_addr}
	else
		docker ps -a | grep -w ${docker_name}
		echo "please delete no use docker container"
	fi
}

function print_help () {
    echo "Usage:
    bash docker.sh [OPTION]... [OPTARGS]..."
    echo "Options:
    -i|     pull the images, del old container, start a new container
    -p|     push commit new container to master2
    -r|     start the docker container
    -n|     start a new docker container terminal
    -s|     stop the docker container
    -d|     delete the docker container
    -h|     print help information
	"
}

# main entry
if [ $# == 0 ]; then
   print_help
   exit -1
fi

while getopts "iprnsdh" arg
do
    case $arg in

        i)
			docker_init
			exit 0
            ;;
        p)
      docker_commit
      exit 0
            ;;
        r)
			docker_run
			exit 0
            ;;
        n)
            docker_new_ternimal
			exit 0
            ;;
        s)
            docker_stop
			exit 0
            ;;
        d)
			docker_del
            exit 0
            ;;
        h)
            print_help
            exit 0
            ;;
    esac
done