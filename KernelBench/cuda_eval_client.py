from typing import Tuple

import json
import time
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ClientConfig:
    server_url: str = "http://localhost:6000"
    timeout: int = 300


class CUDAEvalClient:
    """Client for interacting with CUDA Evaluation Server"""

    def __init__(self, server_url: str = "http://localhost:6000"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.server_url}/")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and device information"""
        response = self.session.get(f"{self.server_url}/status")
        response.raise_for_status()
        return response.json()

    def submit_evaluation(self, problem_id: int, sample_id: int, custom_cuda: str,
                          ref_arch_src: str, level: Optional[int] = None,
                          timeout: Optional[int] = None) -> str:
        """Submit an evaluation job and return job ID"""
        data = {
            "problem_id": problem_id,
            "sample_id": sample_id,
            "custom_cuda": custom_cuda,
            "ref_arch_src": ref_arch_src
        }

        if level is not None:
            data["level"] = level
        if timeout is not None:
            data["timeout"] = timeout

        response = self.session.post(f"{self.server_url}/evaluate", json=data)
        response.raise_for_status()
        return response.json()["job_id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status and results"""
        response = self.session.get(f"{self.server_url}/job/{job_id}")
        response.raise_for_status()
        return response.json()

    def evaluate_sync(self, problem_id: int, sample_id: int, custom_cuda: str,
                      ref_arch_src: str, level: Optional[int] = None,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """Submit evaluation and wait for completion (synchronous)"""
        data = {
            "problem_id": problem_id,
            "sample_id": sample_id,
            "custom_cuda": custom_cuda,
            "ref_arch_src": ref_arch_src
        }

        if level is not None:
            data["level"] = level
        if timeout is not None:
            data["timeout"] = timeout

        response = self.session.post(f"{self.server_url}/evaluate_sync", json=data)
        response.raise_for_status()
        return response.json()

    def wait_for_job(self, job_id: str, timeout: int = 300, poll_interval: float = 1.0) -> Dict[str, Any]:
        """Wait for a job to complete and return the result"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            job_status = self.get_job_status(job_id)

            if job_status["status"] in ["completed", "failed"]:
                return job_status

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def list_jobs(self) -> Dict[str, Any]:
        """List all jobs (active and completed)"""
        response = self.session.get(f"{self.server_url}/jobs")
        response.raise_for_status()
        return response.json()
