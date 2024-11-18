#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# ========== Setup the benchmark tools ========== #

# Clone MultiPL-E repository
if [ ! -d "MultiPL-E" ]; then
    echo "Cloning MultiPL-E repository..."
    git clone https://github.com/nuprl/MultiPL-E.git
    cd MultiPL-E
    git checkout 19a25675e6df678945a6e3da0dca9473265b0055
    cd ..
fi

# ========== Setup the benchmark parameters ========== #

MODEL_NAME=$1
MODEL_LABEL=$(echo $MODEL_NAME | cut -d'/' -f 2)

LANGUAGE="jl"
BENCHMARK_DATASET="humaneval"

BATCH_SIZE=8
MAX_TOKENS=1024
TEMPERATURE=0.2
COMPLETION_LIMIT=1

# Create output directory
RESULTS_DIR="./results/$MODEL_LABEL"
mkdir -p $RESULTS_DIR

OUTPUT_DIR="${RESULTS_DIR}/${LANGUAGE}_benchmark_temperature_${TEMPERATURE}"
mkdir -p $OUTPUT_DIR


# ========== Running model generation ========== #
echo "Running benchmark with the following parameters:"
echo "Model name: $MODEL_NAME, Benchmark dataset: $BENCHMARK_DATASET, Language: $LANGUAGE, Temperature: $TEMPERATURE, Batch size: $BATCH_SIZE, Completion limit: $COMPLETION_LIMIT, Max tokens: $MAX_TOKENS"

# Run the model generation script
echo "Running model generation script..."
python3 -u ./MultiPL-E/automodel.py \
        --name $MODEL_NAME \
        --root-dataset $BENCHMARK_DATASET \
        --lang $LANGUAGE \
        --temperature $TEMPERATURE \
        --batch-size $BATCH_SIZE \
        --completion-limit $COMPLETION_LIMIT \
        --output-dir-prefix $OUTPUT_DIR \
        --max-tokens $MAX_TOKENS 

echo "Model generation completed. Results are saved in $OUTPUT_DIR"
