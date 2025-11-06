"""Class for generating Slurm scripts to run vLLM servers.

This module provides functionality to generate Slurm scripts for running vLLM servers
in both single-node and multi-node configurations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from vec_inf.client._client_vars import SLURM_JOB_CONFIG_ARGS
from vec_inf.client._slurm_templates import (
    BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE,
    BATCH_SLURM_SCRIPT_TEMPLATE,
    SLURM_SCRIPT_TEMPLATE,
)


class SlurmScriptGenerator:
    """A class to generate Slurm scripts for running vLLM servers.

    This class handles the generation of Slurm scripts for both single-node and
    multi-node configurations, supporting different virtualization environments
    (venv or singularity/apptainer).

    Parameters
    ----------
        params : dict[str, Any]
            Configuration parameters for the Slurm script.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.is_multinode = int(self.params["num_nodes"]) > 1
        self.use_container = (
            self.params["venv"] == "singularity" or self.params["venv"] == "apptainer"
        )
        self.additional_binds = self.params.get("bind", "")
        if self.additional_binds:
            self.additional_binds = f" --bind {self.additional_binds}"
        model_weights_path = Path(
            self.params["model_weights_parent_dir"], self.params["model_name"]
        )
        model_weights_path.mkdir(parents=True, exist_ok=True)
        self.model_weights_path = str(model_weights_path)
        env_dict: dict[str, str] = self.params.get("env", {})
        # Create string of environment variables
        self.env_str = ""
        for key, val in env_dict.items():
            if len(self.env_str) == 0:
                self.env_str = "--env "
            else:
                self.env_str += ","
            self.env_str += key + "=" + val
        
        # Get HF cache dir from params (CLI/config) or environment variable
        import os
        self.hf_cache_dir = self.params.get("hf_cache_dir") or os.getenv("HF_CACHE_DIR")

    def _get_env_vars(self) -> list[str]:
        """Get environment variables, including HF cache vars if configured.
        
        Returns
        -------
        list[str]
            List of environment variable export commands
        """
        from vec_inf.client.slurm_vars import LD_LIBRARY_PATH, VLLM_NCCL_SO_PATH
        
        env_vars = [
            f"export LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
            f"export VLLM_NCCL_SO_PATH={VLLM_NCCL_SO_PATH}",
        ]
        
        if self.hf_cache_dir:
            env_vars += [
                f"export HF_HOME={self.hf_cache_dir}",
                f"export TRANSFORMERS_CACHE={self.hf_cache_dir}/transformers",
                "mkdir -p $HF_HOME",
                "mkdir -p $TRANSFORMERS_CACHE",
                "export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES",
                "export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES",
                "export APPTAINERENV_HF_HOME=$HF_HOME",
                "export SINGULARITYENV_HF_HOME=$HF_HOME",
                "export APPTAINERENV_TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE",
                "export SINGULARITYENV_TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE",
            ]
        
        return env_vars

    def _get_container_command(self) -> str:
        """Get the container command with HF cache bind if configured.
        
        Returns
        -------
        str
            Container command string
        """
        from vec_inf.client.slurm_vars import SINGULARITY_IMAGE, APPTAINER_IMAGE
        
        container_type = "singularity" if self.params["venv"] == "singularity" else "apptainer"
        container_image = self.params.get("container_image") or (
            SINGULARITY_IMAGE if container_type == "singularity" else APPTAINER_IMAGE
        )
        is_docker_uri = container_image.startswith("docker://")
        
        # Build base command
        if container_type == "singularity":
            base_cmd = "singularity exec --nv"
        else:
            base_cmd = "apptainer exec --nv"
        
        # Add HF cache bind if configured
        hf_cache_bind = ""
        if self.hf_cache_dir:
            try:
                if Path(self.hf_cache_dir).exists():
                    hf_cache_bind = f" --bind {self.hf_cache_dir}"
            except Exception:
                pass  # Skip bind if path doesn't exist
        
        # Build bind mounts
        binds = f" --bind {self.model_weights_path}{hf_cache_bind}{self.additional_binds}"
        
        # Add containall flag only for .sif files (not Docker URIs)
        containall_flag = "" if is_docker_uri else " --containall"
        
        # Add HF env vars if configured
        env_vars = ""
        if self.hf_cache_dir:
            env_vars = f" --env HF_HOME={self.hf_cache_dir} --env TRANSFORMERS_CACHE={self.hf_cache_dir}/transformers"
        
        return f"{base_cmd}{env_vars}{binds}{containall_flag} {container_image} \\"

    def _generate_script_content(self) -> str:
        """Generate the complete Slurm script content.

        Returns
        -------
        str
            The complete Slurm script as a string.
        """
        script_content = []
        script_content.append(self._generate_shebang())
        script_content.append(self._generate_server_setup())
        script_content.append(self._generate_launch_cmd())
        return "\n".join(script_content)

    def _generate_shebang(self) -> str:
        """Generate the Slurm script shebang with job specifications.

        Returns
        -------
        str
            Slurm shebang containing job specifications.
        """
        shebang = [SLURM_SCRIPT_TEMPLATE["shebang"]["base"]]
        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
        if self.is_multinode:
            shebang += SLURM_SCRIPT_TEMPLATE["shebang"]["multinode"]
        return "\n".join(shebang)

    def _generate_server_setup(self) -> str:
        """Generate the server initialization script.

        Creates the script section that handles server setup, including Ray
        initialization for multi-node setups and port configuration.

        Returns
        -------
        str
            Server initialization script content.
        """
        server_script = ["\n"]
        if self.use_container:
            server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["container_setup"]))
        server_script.append("\n".join(self._get_env_vars()))
        server_script.append(
            SLURM_SCRIPT_TEMPLATE["imports"].format(src_dir=self.params["src_dir"])
        )
        if self.is_multinode:
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["multinode"]
            ).format(gpus_per_node=self.params["gpus_per_node"])
            if self.use_container:
                server_setup_str = server_setup_str.replace(
                    "CONTAINER_PLACEHOLDER",
                    self._get_container_command(),
                )
        else:
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["single_node"]
            )
        server_script.append(server_setup_str)
        server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["find_vllm_port"]))
        server_script.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["write_to_json"]).format(
                log_dir=self.params["log_dir"], model_name=self.params["model_name"]
            )
        )
        return "\n".join(server_script)

    def _generate_launch_cmd(self) -> str:
        """Generate the vLLM server launch command.

        Creates the command to launch the vLLM server, handling different virtualization
        environments (venv or singularity/apptainer).

        Returns
        -------
        str
            Server launch command.
        """
        launcher_script = ["\n"]

        # Check if --model is specified in vllm_args to use HuggingFace model name
        model_path = self.model_weights_path
        vllm_args_copy = self.params["vllm_args"].copy()
        if "--model" in vllm_args_copy:
            model_path = vllm_args_copy.pop("--model")

        if self.use_container:
            launcher_script.append(self._get_container_command())
        else:
            launcher_script.append(
                SLURM_SCRIPT_TEMPLATE["activate_venv"].format(venv=self.params["venv"])
            )
        launcher_script.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["launch_cmd"]).format(
                model_weights_path=model_path,
                model_name=self.params["model_name"],
            )
        )

        for arg, value in vllm_args_copy.items():
            if isinstance(value, bool):
                launcher_script.append(f"    {arg} \\")
            else:
                launcher_script.append(f"    {arg} {value} \\")
        return "\n".join(launcher_script)

    def write_to_log_dir(self) -> Path:
        """Write the generated Slurm script to the log directory.

        Creates a timestamped script file in the configured log directory.

        Returns
        -------
        Path
            Path to the generated Slurm script file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path: Path = (
            Path(self.params["log_dir"])
            / f"launch_{self.params['model_name']}_{timestamp}.sbatch"
        )

        content = self._generate_script_content()
        script_path.write_text(content)
        return script_path


class BatchSlurmScriptGenerator:
    """A class to generate Slurm scripts for batch mode.

    This class handles the generation of Slurm scripts for batch mode, which
    launches multiple vLLM servers with different configurations in parallel.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.script_paths: list[Path] = []
        self.use_container = (
            self.params["venv"] == "singularity" or self.params["venv"] == "apptainer"
        )
        # Get HF cache dir from params (CLI/config) or environment variable
        import os
        self.hf_cache_dir = self.params.get("hf_cache_dir") or os.getenv("HF_CACHE_DIR")
        for model_name in self.params["models"]:
            self.params["models"][model_name]["additional_binds"] = ""
            if self.params["models"][model_name].get("bind"):
                self.params["models"][model_name]["additional_binds"] = (
                    f" --bind {self.params['models'][model_name]['bind']}"
                )
            model_weights_path = Path(
                self.params["models"][model_name]["model_weights_parent_dir"],
                model_name,
            )
            model_weights_path.mkdir(parents=True, exist_ok=True)
            self.params["models"][model_name]["model_weights_path"] = str(
                model_weights_path
            )

    def _get_batch_env_vars(self) -> list[str]:
        """Get environment variables for batch mode, including HF cache vars if configured.
        
        Returns
        -------
        list[str]
            List of environment variable export commands
        """
        from vec_inf.client.slurm_vars import LD_LIBRARY_PATH, VLLM_NCCL_SO_PATH
        
        env_vars = [
            f"export LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
            f"export VLLM_NCCL_SO_PATH={VLLM_NCCL_SO_PATH}",
        ]
        
        if self.hf_cache_dir:
            env_vars += [
                f"export HF_HOME={self.hf_cache_dir}",
                f"export TRANSFORMERS_CACHE={self.hf_cache_dir}/transformers",
                "mkdir -p $HF_HOME",
                "mkdir -p $TRANSFORMERS_CACHE",
                "export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES",
                "export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES",
                "export APPTAINERENV_HF_HOME=$HF_HOME",
                "export SINGULARITYENV_HF_HOME=$HF_HOME",
                "export APPTAINERENV_TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE",
                "export SINGULARITYENV_TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE",
            ]
        
        return env_vars

    def _get_batch_container_command(self, model_weights_path: str, additional_binds: str) -> str:
        """Get the container command for batch mode with HF cache bind if configured.
        
        Parameters
        ----------
        model_weights_path : str
            Path to model weights
        additional_binds : str
            Additional bind mounts
            
        Returns
        -------
        str
            Container command string
        """
        from vec_inf.client.slurm_vars import SINGULARITY_IMAGE, APPTAINER_IMAGE
        
        container_type = "singularity" if self.params["venv"] == "singularity" else "apptainer"
        container_image = self.params.get("container_image") or (
            SINGULARITY_IMAGE if container_type == "singularity" else APPTAINER_IMAGE
        )
        is_docker_uri = container_image.startswith("docker://")
        
        # Build base command
        if container_type == "singularity":
            base_cmd = "singularity exec --nv"
        else:
            base_cmd = "apptainer exec --nv"
        
        # Add HF cache bind if configured
        hf_cache_bind = ""
        if self.hf_cache_dir:
            try:
                if Path(self.hf_cache_dir).exists():
                    hf_cache_bind = f" --bind {self.hf_cache_dir}"
            except Exception:
                pass  # Skip bind if path doesn't exist
        
        # Build bind mounts
        binds = f" --bind {model_weights_path}{hf_cache_bind}{additional_binds}"
        
        # Add containall flag only for .sif files (not Docker URIs)
        containall_flag = "" if is_docker_uri else " --containall"
        
        # Add HF env vars if configured
        env_vars = ""
        if self.hf_cache_dir:
            env_vars = f" --env HF_HOME={self.hf_cache_dir} --env TRANSFORMERS_CACHE={self.hf_cache_dir}/transformers"
        
        return f"{base_cmd}{env_vars}{binds}{containall_flag} {container_image} \\"

    def _write_to_log_dir(self, script_content: list[str], script_name: str) -> Path:
        """Write the generated Slurm script to the log directory.

        Returns
        -------
        Path
            The Path object to the generated Slurm script file.
        """
        script_path = Path(self.params["log_dir"]) / script_name
        script_path.touch(exist_ok=True)
        script_path.write_text("\n".join(script_content))
        return script_path

    def _generate_model_launch_script(self, model_name: str) -> Path:
        """Generate the bash script for launching individual vLLM servers.

        Parameters
        ----------
        model_name : str
            The name of the model to launch.

        Returns
        -------
        Path
            The bash script path for launching the vLLM server.
        """
        # Generate the bash script content
        script_content = []
        model_params = self.params["models"][model_name]
        script_content.append(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["shebang"])
        script_content.append("\n".join(self._get_batch_env_vars()))
        script_content.append(
            "\n".join(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["server_address_setup"]
            ).format(src_dir=self.params["src_dir"])
        )
        script_content.append(
            "\n".join(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["write_to_json"]).format(
                het_group_id=model_params["het_group_id"],
                log_dir=self.params["log_dir"],
                slurm_job_name=self.params["slurm_job_name"],
                model_name=model_name,
            )
        )
        # Check if --model is specified in vllm_args to use HuggingFace model name
        model_path = model_params["model_weights_path"]
        vllm_args_copy = model_params["vllm_args"].copy()
        if "--model" in vllm_args_copy:
            model_path = vllm_args_copy.pop("--model")

        if self.use_container:
            script_content.append(
                self._get_batch_container_command(
                    model_params["model_weights_path"],
                    model_params["additional_binds"],
                )
            )
        script_content.append(
            "\n".join(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["launch_cmd"]).format(
                model_weights_path=model_path,
                model_name=model_name,
            )
        )
        
        for arg, value in vllm_args_copy.items():
            if isinstance(value, bool):
                script_content.append(f"    {arg} \\")
            else:
                script_content.append(f"    {arg} {value} \\")
        script_content[-1] = script_content[-1].replace("\\", "")
        # Write the bash script to the log directory
        launch_script_path = self._write_to_log_dir(
            script_content, f"launch_{model_name}.sh"
        )
        self.script_paths.append(launch_script_path)
        return launch_script_path

    def _generate_batch_slurm_script_shebang(self) -> str:
        """Generate the shebang for batch mode Slurm script.

        Returns
        -------
        str
            The shebang for batch mode Slurm script.
        """
        shebang = [BATCH_SLURM_SCRIPT_TEMPLATE["shebang"]]

        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
        shebang.append("#SBATCH --ntasks=1")
        shebang.append("\n")

        for model_name in self.params["models"]:
            shebang.append(f"# ===== Resource group for {model_name} =====")
            for arg, value in SLURM_JOB_CONFIG_ARGS.items():
                model_params = self.params["models"][model_name]
                if model_params.get(value) and value not in ["out_file", "err_file"]:
                    shebang.append(f"#SBATCH --{arg}={model_params[value]}")
            shebang[-1] += "\n"
            shebang.append(BATCH_SLURM_SCRIPT_TEMPLATE["hetjob"])
        # Remove the last hetjob line
        shebang.pop()
        return "\n".join(shebang)

    def generate_batch_slurm_script(self) -> Path:
        """Generate the Slurm script for launching multiple vLLM servers in batch mode.

        Returns
        -------
        Path
            The Slurm script for launching multiple vLLM servers in batch mode.
        """
        script_content = []

        script_content.append(self._generate_batch_slurm_script_shebang())

        for model_name in self.params["models"]:
            model_params = self.params["models"][model_name]
            script_content.append(f"# ===== Launching {model_name} =====")
            launch_script_path = str(self._generate_model_launch_script(model_name))
            script_content.append(
                BATCH_SLURM_SCRIPT_TEMPLATE["permission_update"].format(
                    script_name=launch_script_path
                )
            )
            script_content.append(
                "\n".join(BATCH_SLURM_SCRIPT_TEMPLATE["launch_model_scripts"]).format(
                    het_group_id=model_params["het_group_id"],
                    out_file=model_params["out_file"],
                    err_file=model_params["err_file"],
                    script_name=launch_script_path,
                )
            )
        script_content.append("wait")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"{self.params['slurm_job_name']}_{timestamp}.sbatch"
        return self._write_to_log_dir(script_content, script_name)