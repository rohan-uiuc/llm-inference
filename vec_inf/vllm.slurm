#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Load CUDA modules in correct order
module purge
module load gcc/11.3.0
module load cuda/11.7.0
module load cudnn/8.4.0.27-11.6
#module load anaconda3_gpu/4.13.0
# module load cuda-12.2.1

# Export GCC paths to help compiler find it in containers
export GCC_PATH=/sw/spack/hydrogpu-2022-06/apps/gcc/11.3.0-gcc-8.4.1-kroofxa
export PATH=$GCC_PATH/bin:$PATH
export LD_LIBRARY_PATH=$GCC_PATH/lib64:$LD_LIBRARY_PATH

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=0  # For single GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES=$DEVICES
fi

# Verify CUDA setup
nvidia-smi

# Set HuggingFace cache directory to use the model weights directory
export HF_HOME="${MODEL_WEIGHTS_PARENT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${MODEL_WEIGHTS_PARENT_DIR}/.cache/huggingface/transformers"
mkdir -p "${HF_HOME}"
mkdir -p "${TRANSFORMERS_CACHE}"

source ${SRC_DIR}/find_port.sh

# Write server url to file
hostname=${SLURMD_NODENAME}
vllm_port_number=$(find_available_port $hostname 8080 65535)

echo "Server address: http://${hostname}:${vllm_port_number}/v1"

# Setup Cloudflare tunnel if enabled
if [ "$ENABLE_CLOUDFLARE_TUNNEL" = "True" ]; then
    # Get the SLURM job ID
    job_id=${SLURM_JOB_ID}
    
    # Create Cloudflare tunnel
    tunnel_url=$(bash ${SRC_DIR}/cloudflare_tunnel.sh create ${job_id} ${hostname} ${vllm_port_number})
    
    # Save only the last line (the actual URL) to a file for later reference
    echo "${tunnel_url}" | tail -n 1 > "${LOG_DIR}/${JOB_NAME}.${job_id}.tunnel_url"
    
    echo "Cloudflare tunnel created. Service available at: ${tunnel_url}"
fi

if [ "$ENFORCE_EAGER" = "True" ]; then
    export ENFORCE_EAGER="--enforce-eager"
else
    export ENFORCE_EAGER=""
fi

# Set model path - use HuggingFace ID if provided, otherwise use local path
if [ -n "$HUGGINGFACE_ID" ]; then
    export MODEL_PATH="$HUGGINGFACE_ID"
else
    export MODEL_PATH="$VLLM_MODEL_WEIGHTS"
fi

echo "VENV_BASE: $VENV_BASE"
echo "Using model path: $MODEL_PATH"

# Activate vllm venv
if [ "$VENV_BASE" = "singularity" ]; then
    export SINGULARITY_IMAGE=/projects/bbwb/rohan13/llm-inference/vector-inference.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    module load singularity-ce/3.8.2
    
    # Setup container environment
    singularity exec --nv \
    --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} \
    --bind ${GCC_PATH}:${GCC_PATH} \
    --bind ${SRC_DIR}:${SRC_DIR} \
    $SINGULARITY_IMAGE \
    bash ${SRC_DIR}/container_setup.sh
    
    # Stop any existing Ray processes
    singularity exec $SINGULARITY_IMAGE ray stop
    
    # Run the main vLLM server
    singularity exec --nv \
    --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} \
    --bind ${GCC_PATH}:${GCC_PATH} \
    --bind ${SRC_DIR}:${SRC_DIR} \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env PATH=$PATH \
    --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
elif [ "$VENV_BASE" = "apptainer" ]; then
    export SINGULARITY_IMAGE=/projects/bbwb/rohan13/llm-inference/vector-inference.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    
    # Setup container environment
    apptainer exec --nv \
    --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} \
    --bind ${GCC_PATH}:${GCC_PATH} \
    --bind ${SRC_DIR}:${SRC_DIR} \
    $SINGULARITY_IMAGE \
    bash ${SRC_DIR}/container_setup.sh
    
    # Run the main vLLM server
    apptainer exec --nv \
    --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} \
    --bind ${GCC_PATH}:${GCC_PATH} \
    --bind ${SRC_DIR}:${SRC_DIR} \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env PATH=$PATH \
    --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
else
    source ${VENV_BASE}/bin/activate
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
fi
