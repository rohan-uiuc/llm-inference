"""
Script for reporting OpenAI base URL and model info to e.g., KScope when ready.
"""

from typing import Any

import argparse
import logging
import os
import requests
import time

parser = argparse.ArgumentParser()
parser.add_argument("--openai_base_url")
parser.add_argument("--telemetry_callback_url")


def get_openai_model_list(openai_base_url: str) -> list[dict[str, Any]] | None:
    """
    Return model list when the given OpenAI-compatible API is ready.

    This function synchronously waits on the HTTP request.

    Params:
        openai_base_url: Full Path of the OpenAI URL, including "/v1"
            e.g., "http://localhost:8080/v1"

    Returns:
        Model list from the OpenAI API at the given base URL.
        Model list might be empty.

        None if API response code >= 400.
    """
    model_list_url = "{}/{}".format(openai_base_url.rstrip("/"), "models")

    try:
        response: requests.Response = requests.get(model_list_url)
    # connection refused, etc.
    except requests.exceptions.ConnectionError:
        return None

    if response.ok:
        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError:
            return None

        assert isinstance(response_json, dict), response_json
        assert "data" in response_json, response_json
        return response_json["data"]

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    logger = logging.getLogger("vector_inference_callback")
    logging.basicConfig(level=logging.INFO)

    openai_base_url = args.openai_base_url
    telemetry_callback_url = args.telemetry_callback_url

    model_list = []

    while (model_list is None) or (len(model_list) == 0):
        model_list = get_openai_model_list(openai_base_url)
        logger.info(f"model_list: {model_list}")
        time.sleep(10)

    telemetry_data = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_list": model_list,
        "api_base_url": openai_base_url,
    }
    logger.info(f"sending telemetry to {telemetry_callback_url}: {telemetry_data}")
    requests.post(telemetry_callback_url, json=telemetry_data)
