#! bin/bash
if [[ "$1" != "--ignore" ]]; then
    rm /home/adam/milvus/cmake_build/bin/geos-config
    make -j8
fi
pushd deployments/docker/dev/
docker compose down
sudo rm -rf ./volumes/*
docker compose up -d
popd
