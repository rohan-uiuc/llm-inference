"""Command line interface for Vector Inference."""

import json
import os
from typing import Any, Optional, Union, cast

import click
from rich.console import Console
from rich.table import Table
import glob

import vec_inf.cli._utils as utils
from vec_inf.cli._config import ModelConfig


class LaunchHelper:
    def __init__(
        self, model_name: str, cli_kwargs: dict[str, Optional[Union[str, int, bool]]]
    ):
        self.model_name = model_name
        self.cli_kwargs = cli_kwargs
        self.model_config = self.get_model_configuration()

    def get_model_configuration(self) -> ModelConfig:
        """Load and validate model configuration."""
        model_configs = utils.load_config()
        if config := next(
            (m for m in model_configs if m.model_name == self.model_name), None
        ):
            return config
        raise click.ClickException(
            f"Model '{self.model_name}' not found in configuration"
        )

    def get_base_launch_command(self) -> str:
        """Construct base launch command."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "launch_server.sh",
        )
        return f"bash {script_path}"

    def process_configuration(self) -> dict[str, Any]:
        """Merge config defaults with CLI overrides."""
        params = self.model_config.model_dump(exclude={"model_name"})

        # Process boolean fields
        for bool_field in ["pipeline_parallelism", "enforce_eager"]:
            if (value := self.cli_kwargs.get(bool_field)) is not None:
                params[bool_field] = self.convert_boolean_value(value)
        
        # Handle flag options
        if self.cli_kwargs.get("enable_cloudflare_tunnel"):
            params["enable_cloudflare_tunnel"] = "True"

        # Merge other overrides
        for key, value in self.cli_kwargs.items():
            if value is not None and key not in [
                "json_mode",
                "pipeline_parallelism",
                "enforce_eager",
                "enable_cloudflare_tunnel",
            ]:
                params[key] = value
        return params

    def convert_boolean_value(self, value: Union[str, int, bool]) -> str:
        """Convert various input types to boolean strings."""
        if isinstance(value, str):
            return "True" if value.lower() == "true" else "False"
        return "True" if bool(value) else "False"

    def build_launch_command(self, base_command: str, params: dict[str, Any]) -> str:
        """Construct the full launch command with parameters."""
        print(f"Building launch command with params: {params}")
        command = base_command
        for param_name, param_value in params.items():
            if param_value is None:
                continue

            formatted_value = param_value
            if isinstance(formatted_value, bool):
                formatted_value = "True" if formatted_value else "False"

            arg_name = param_name.replace("_", "-")
            
            # Handle flag options
            if param_name == "enable_cloudflare_tunnel" and param_value == "True":
                command += f" --{arg_name}"
                continue
                
            command += f" --{arg_name} {formatted_value}"

        return command

    def parse_launch_output(self, output: str) -> tuple[str, list[str]]:
        """Extract job ID and output lines from command output."""
        slurm_job_id = output.split(" ")[-1].strip().strip("\n")
        output_lines = output.split("\n")[:-2]
        return slurm_job_id, output_lines

    def format_json_output(self, job_id: str, lines: list[str]) -> str:
        """Format output as JSON string with proper double quotes."""
        output_data = {"slurm_job_id": job_id}
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                output_data[key.lower().replace(" ", "_")] = value
        return json.dumps(output_data)

    def format_table_output(self, job_id: str, lines: list[str]) -> Table:
        """Format output as rich Table."""
        table = utils.create_table(key_title="Job Config", value_title="Value")
        table.add_row("Slurm Job ID", job_id, style="blue")
        for line in lines:
            key, value = line.split(": ")
            table.add_row(key, value)
        return table

    def handle_launch_output(self, output: str, console: Console) -> None:
        """Process and display launch output."""
        json_mode = bool(self.cli_kwargs.get("json_mode", False))
        slurm_job_id, output_lines = self.parse_launch_output(output)
        console.print(f"SLURM Job ID: {slurm_job_id}")
        console.print(f"Output lines: {output_lines}")

        if json_mode:
            output_data = self.format_json_output(slurm_job_id, output_lines)
            click.echo(output_data)
        else:
            table = self.format_table_output(slurm_job_id, output_lines)
            console.print(table)


class StatusHelper:
    def __init__(self, slurm_job_id: int, output: str, log_dir: Optional[str] = None):
        self.slurm_job_id = slurm_job_id
        self.output = output
        self.log_dir = log_dir if log_dir else os.path.join(os.getcwd(), "logs")
        self.status_info = self.get_base_status_data()

    def get_base_status_data(self) -> dict[str, Union[str, None]]:
        """Extract basic job status information from scontrol output."""
        try:
            print(f"get_base_status_data Output: {self.output}")
            job_name = self.output.split(" ")[1].split("=")[1]
            job_state = self.output.split(" ")[9].split("=")[1]
        except IndexError:
            job_name = "UNAVAILABLE"
            job_state = "UNAVAILABLE"

        return {
            "model_name": job_name,
            "status": "UNAVAILABLE",
            "base_url": "UNAVAILABLE",
            "cloudflare_url": "UNAVAILABLE",
            "state": job_state,
            "pending_reason": None,
            "failed_reason": None,
        }

    def process_job_state(self) -> None:
        """Process job state and update status information."""
        if self.status_info["state"] == "RUNNING":
            self.process_running_state()
        elif self.status_info["state"] == "PENDING":
            self.process_pending_state()
        elif self.status_info["state"] in ["CANCELLED", "COMPLETED", "FAILED", "TIMEOUT"]:
            self.status_info["status"] = "SHUTDOWN"
        else:
            self.status_info["status"] = "UNAVAILABLE"

    def check_model_health(self) -> None:
        """Check if the model is healthy by reading the log file."""
        try:
            log_file = glob.glob(
                f"{self.log_dir}/{self.status_info['model_name']}.{self.slurm_job_id}.out"
            )[0]
            with open(log_file, "r") as f:
                log_content = f.read()

            if "Server address:" in log_content:
                self.status_info["status"] = "READY"
                server_url_line = [
                    line for line in log_content.split("\n") if "Server address:" in line
                ][0]
                self.status_info["base_url"] = server_url_line.split("Server address: ")[
                    1
                ]
                
                # Check for Cloudflare tunnel URL
                tunnel_url_file = f"{self.log_dir}/{self.status_info['model_name']}.{self.slurm_job_id}.tunnel_url"
                if os.path.exists(tunnel_url_file):
                    with open(tunnel_url_file, "r") as f:
                        # Read all lines and get the last non-empty line which should be just the URL
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        if lines:
                            self.status_info["cloudflare_url"] = lines[-1]
            else:
                self.status_info["status"] = "LAUNCHING"
        except (IndexError, FileNotFoundError):
            self.status_info["status"] = "LAUNCHING"

    def process_running_state(self) -> None:
        """Process running state and check model health."""
        self.check_model_health()
        if self.status_info["status"] == "LAUNCHING":
            # Check if the job has been running for too long without becoming ready
            try:
                log_file = glob.glob(
                    f"{self.log_dir}/{self.status_info['model_name']}.{self.slurm_job_id}.err"
                )[0]
                with open(log_file, "r") as f:
                    err_content = f.read()
                if "error" in err_content.lower() or "exception" in err_content.lower():
                    self.status_info["status"] = "FAILED"
                    self.status_info["failed_reason"] = "Error in model initialization"
            except (IndexError, FileNotFoundError):
                pass

    def process_pending_state(self) -> None:
        """Process pending state and extract pending reason."""
        self.status_info["status"] = "PENDING"
        try:
            reason_index = self.output.find("Reason=")
            if reason_index != -1:
                reason_str = self.output[reason_index:]
                end_index = reason_str.find(" ")
                if end_index != -1:
                    self.status_info["pending_reason"] = reason_str[
                        len("Reason=") : end_index
                    ]
        except Exception:
            pass

    def output_json(self) -> None:
        """Output status information as JSON."""
        output_data = {
            "slurm_job_id": self.slurm_job_id,
            "model_name": self.status_info["model_name"],
            "status": self.status_info["status"],
            "base_url": self.status_info["base_url"],
            "cloudflare_url": self.status_info["cloudflare_url"],
        }
        if self.status_info["pending_reason"]:
            output_data["pending_reason"] = self.status_info["pending_reason"]
        if self.status_info["failed_reason"]:
            output_data["failed_reason"] = self.status_info["failed_reason"]
        click.echo(json.dumps(output_data))

    def output_table(self, console: Console) -> None:
        """Output status information as a table."""
        table = Table(title=f"Model Status (Job ID: {self.slurm_job_id})")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Model Name", self.status_info["model_name"])
        table.add_row("Status", self.status_info["status"])
        table.add_row("Base URL", self.status_info["base_url"])
        table.add_row("Cloudflare URL", self.status_info["cloudflare_url"])
        if self.status_info["pending_reason"]:
            table.add_row("Pending Reason", self.status_info["pending_reason"])
        if self.status_info["failed_reason"]:
            table.add_row("Failed Reason", self.status_info["failed_reason"])
        console.print(table)
