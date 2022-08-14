CODE_DIR="./zemi"
DATA_DIR="./data"
RETRIEVAL_DIR="./c4_retrieval"

IMAGE_NAME="mirrors.tencent.com/ai-lab-seattle/mikeeewang_t0"

docker run --ipc=host --network=host --rm -it --gpus=all \
    --privileged=true \
    --name="zsemi_container" \
    -v $CODE_DIR:/code \
    -v $DATA_DIR:/data \
    -v $RETRIEVAL_DIR:/c4_retrieval \
    -w /code \
    ${IMAGE_NAME} /bin/bash
    # -v $CACHE_DIR:/cache \
