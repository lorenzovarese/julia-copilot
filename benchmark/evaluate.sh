#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ========== Setup the benchmark tools ========== #

# Clone MultiPL-E repository
if [ ! -d "MultiPL-E" ]; then
    echo "Cloning MultiPL-E repository..."
    git clone https://github.com/nuprl/MultiPL-E.git
    cd MultiPL-E
    git checkout 19a25675e6df678945a6e3da0dca9473265b0055
    cd ..
fi

# Setup MultiPL-E docker image
TAG="multipl-e-eval"
IMAGE="ghcr.io/nuprl/multipl-e-evaluation:latest"
if [ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]; then
    echo "Pulling and tagging Multipl-E image..."
    docker pull $IMAGE
    docker tag $IMAGE $TAG
else
    echo "Multipl-E image already set up. Continuing..."
fi

# ========== Setup the benchmark parameters ========== #

MODEL_NAME=$1
MODEL_LABEL=$(echo $MODEL_NAME | cut -d'/' -f 2)

LANGUAGE="jl"
TEMPERATURE=0.2

RESULTS_DIR="./results/$MODEL_LABEL"
OUTPUT_DIR="${RESULTS_DIR}/${LANGUAGE}_benchmark_temperature_${TEMPERATURE}"

# ========== Running model evaluation ========== #
echo "Running evaluation script..."
docker run --rm --network none -v ./$OUTPUT_DIR:/$OUTPUT_DIR:rw multipl-e-eval --dir /$OUTPUT_DIR --output-dir /$OUTPUT_DIR --recursive

echo "Exporting the predictions in a JSON file..."
python3 -u extract.py -i ./$OUTPUT_DIR -o ${MODEL_LABEL}_results_${LANGUAGE}.json

# Save the results
echo "Printing the results..."
TEMPERATURE_LABEL=$(echo $TEMPERATURE | sed 's/\./_/g')
python3 -u ./MultiPL-E/pass_k.py ./$OUTPUT_DIR/* | tee $OUTPUT_DIR/pass_result_temp_${TEMPERATURE_LABEL}.csv
