"""Slurm cluster configuration variables."""

from pathlib import Path
import os

from typing_extensions import Literal

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


CACHED_CONFIG = Path("/", "model-weights", "vec-inf-shared", "models_latest.yaml")
LD_LIBRARY_PATH = "/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"
# Default container images - use official vLLM Docker image as fallback
SINGULARITY_IMAGE = "docker://vllm/vllm-openai:latest"
SINGULARITY_LOAD_CMD = "# Singularity available at /usr/bin/singularity"
APPTAINER_IMAGE = "docker://vllm/vllm-openai:latest"
APPTAINER_LOAD_CMD = "# Apptainer available at /usr/bin/apptainer"
VLLM_NCCL_SO_PATH = "/vec-inf/nccl/libnccl.so.2.18.1"
MAX_GPUS_PER_NODE = 8
MAX_NUM_NODES = 16
MAX_CPUS_PER_TASK = 128

QOS = Literal[
    "normal",
    "m",
    "m2",
    "m3",
    "m4",
    "m5",
    "long",
    "deadline",
    "high",
    "scavenger",
    "llm",
    "a100",
]

PARTITION = Literal[
    "a40",
    "a100",
    "t4v1",
    "t4v2",
    "rtx6000",
]

DEFAULT_ARGS = {
    "cpus_per_task": 16,
    "mem_per_node": "64G",
    "qos": "m2",
    "time": "08:00:00",
    "partition": "a40",
    "data_type": "auto",
    "log_dir": "~/.vec-inf-logs",
    "model_weights_parent_dir": "/model-weights",
}

# Optional host-side HuggingFace cache directory to bind into containers.
# If set, code will forward HF cache env vars and bind this path conditionally.
# Can be set via environment variable HF_CACHE_DIR or .env file.
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
