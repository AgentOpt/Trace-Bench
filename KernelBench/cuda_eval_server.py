#!/usr/bin/env python3
"""
CUDA Evaluation Server - FastAPI server for managing CUDA kernel evaluations
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import KernelExecResult
try:
    from src.eval import KernelExecResult
except ImportError:
    # Fallback definition if import fails
    @dataclass
    class KernelExecResult:
        compiled: bool = False
        correctness: bool = False
        metadata: dict = None
        runtime: float = -1.0
        runtime_stats: dict = None

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
            if self.runtime_stats is None:
                self.runtime_stats = {}


import subprocess
import sys

REPO_EXTERNAL_LIB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../external/",
    )
)

def invoke_eval_with_subprocess_list(problem_id=1, sample_id=0, custom_cuda=None, ref_arch_src=None,
                                     device="cuda:0", level=None, verbose=False):
    # Use the script path from the attached file
    script_path = os.path.join(REPO_EXTERNAL_LIB_PATH, "scripts/eval_single_example.py")
    args = [
        sys.executable,
        script_path,
        "--problem-id", str(problem_id),
        "--sample-id", str(sample_id),
        "--device", device,
        "--custom-cuda", custom_cuda or "// Default kernel\n__global__ void default_kernel() { }",
        "--ref-arch-src", ref_arch_src or "import torch\nclass DefaultModel(torch.nn.Module): pass"
    ]

    if level is not None:
        args.extend(["--level", str(level)])

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/home/ubuntu/KernelBench"
        )

        # Parse JSON result from output
        import json
        import re

        stdout = result.stdout
        json_start_marker = "=== KERNEL_EXEC_RESULT_JSON ==="
        json_end_marker = "=== END_KERNEL_EXEC_RESULT_JSON ==="

        if json_start_marker in stdout and json_end_marker in stdout:
            start_idx = stdout.find(json_start_marker) + len(json_start_marker)
            end_idx = stdout.find(json_end_marker)
            json_str = stdout[start_idx:end_idx].strip()

            try:
                result_data = json.loads(json_str)

                # Create a simple object with the data
                class SimpleResult:
                    def __init__(self, data):
                        self.compiled = data.get("compiled", False)
                        self.correctness = data.get("correctness", False)
                        self.metadata = data.get("metadata", {})
                        self.runtime = data.get("runtime", -1.0)
                        self.runtime_stats = data.get("runtime_stats", {})

                return SimpleResult(result_data)
            except json.JSONDecodeError:
                pass

        # Fallback result
        class ErrorResult:
            def __init__(self, error_msg):
                self.compiled = False
                self.correctness = False
                self.metadata = {"error": error_msg, "return_code": result.returncode}
                self.runtime = -1.0
                self.runtime_stats = {}

        return ErrorResult(f"Failed to parse result. Return code: {result.returncode}")

    except Exception as e:
        class ExceptionResult:
            def __init__(self, error_msg):
                self.compiled = False
                self.correctness = False
                self.metadata = {"error": error_msg}
                self.runtime = -1.0
                self.runtime_stats = {}

        return ExceptionResult(str(e))


def kernel_exec_result_to_dict(result) -> dict:
    """
    Convert KernelExecResult to a JSON-serializable dictionary
    Handles different types of KernelExecResult objects (dataclass, Pydantic model, etc.)
    """
    if result is None:
        return None

    # Try different methods to convert to dict
    try:
        # Method 1: If it's already a dict
        if isinstance(result, dict):
            return result

        # Method 2: If it has a dict() method (Pydantic model)
        if hasattr(result, 'dict') and callable(getattr(result, 'dict')):
            return result.dict()

        # Method 3: If it's a dataclass
        if hasattr(result, '__dataclass_fields__'):
            return asdict(result)

        # Method 4: Manual extraction (fallback)
        return {
            "compiled": getattr(result, 'compiled', False),
            "correctness": getattr(result, 'correctness', False),
            "metadata": getattr(result, 'metadata', {}),
            "runtime": getattr(result, 'runtime', -1.0),
            "runtime_stats": getattr(result, 'runtime_stats', {})
        }
    except Exception as e:
        logger.error(f"Error converting KernelExecResult to dict: {e}")
        # Return a basic error result
        return {
            "compiled": False,
            "correctness": False,
            "metadata": {"conversion_error": str(e)},
            "runtime": -1.0,
            "runtime_stats": {}
        }


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class EvaluationRequest(BaseModel):
    problem_id: int
    sample_id: int
    custom_cuda: str
    ref_arch_src: str
    level: Optional[int] = None
    timeout: Optional[int] = 300  # seconds


class EvaluationResponse(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    result: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    device: Optional[str] = None
    queued_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class EvaluationJob:
    job_id: str
    request: EvaluationRequest
    status: str = "queued"
    result: Optional[KernelExecResult] = None
    error: Optional[str] = None
    device: Optional[str] = None
    queued_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class CUDADeviceManager:
    """Manages CUDA device allocation and job queue"""

    def __init__(self, cuda_devices: List[str]):
        self.cuda_devices = cuda_devices
        self.available_devices = queue.Queue()
        self.device_status = {}

        # Initialize available devices
        for device in cuda_devices:
            self.available_devices.put(device)
            self.device_status[device] = "idle"

        self.job_queue = queue.Queue()
        self.active_jobs = {}  # job_id -> EvaluationJob
        self.completed_jobs = {}  # job_id -> EvaluationJob
        self.job_counter = 0
        self.lock = threading.Lock()

        # Start worker threads
        self.executor = ThreadPoolExecutor(max_workers=len(cuda_devices))
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info(f"Initialized CUDA Device Manager with devices: {cuda_devices}")

    def submit_job(self, request: EvaluationRequest) -> str:
        """Submit a new evaluation job and return job ID"""
        with self.lock:
            self.job_counter += 1
            job_id = f"job_{self.job_counter}_{int(time.time())}"

        job = EvaluationJob(
            job_id=job_id,
            request=request,
            queued_at=time.time()
        )

        self.active_jobs[job_id] = job
        self.job_queue.put(job)

        logger.info(f"Submitted job {job_id} for problem {request.problem_id}, sample {request.sample_id}")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[EvaluationJob]:
        """Get job status by job ID"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None

    def get_device_status(self) -> Dict[str, str]:
        """Get status of all CUDA devices"""
        return self.device_status.copy()

    def get_queue_info(self) -> Dict[str, int]:
        """Get queue information"""
        return {
            "queued_jobs": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "available_devices": self.available_devices.qsize()
        }

    def _worker_loop(self):
        """Main worker loop that processes jobs"""
        while True:
            try:
                # Get next job from queue
                job = self.job_queue.get(timeout=1.0)

                # Get available device (this will block if no devices available)
                device = self.available_devices.get()

                # Update job and device status
                job.device = device
                job.status = "running"
                job.started_at = time.time()
                self.device_status[device] = "busy"

                logger.info(f"Starting job {job.job_id} on device {device}")

                # Submit to thread pool
                future = self.executor.submit(self._execute_job, job)

                # Handle completion in a separate thread to avoid blocking
                threading.Thread(
                    target=self._handle_job_completion,
                    args=(job, future),
                    daemon=True
                ).start()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")

    def _execute_job(self, job: EvaluationJob) -> None:
        """Execute a single evaluation job"""
        try:
            # Call the evaluation function
            result = invoke_eval_with_subprocess_list(
                problem_id=job.request.problem_id,
                sample_id=job.request.sample_id,
                custom_cuda=job.request.custom_cuda,
                ref_arch_src=job.request.ref_arch_src,
                device=job.device,
                level=job.request.level,
                verbose=False  # Don't print verbose output in server
            )

            job.result = result
            job.status = "completed"

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.error = str(e)
            job.status = "failed"

    def _handle_job_completion(self, job: EvaluationJob, future):
        """Handle job completion and cleanup"""
        try:
            # Wait for job to complete
            future.result()

            # Update completion time
            job.completed_at = time.time()

            # Release device
            self.available_devices.put(job.device)
            self.device_status[job.device] = "idle"

            # Move job to completed
            with self.lock:
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                self.completed_jobs[job.job_id] = job

            logger.info(f"Completed job {job.job_id} on device {job.device} (status: {job.status})")

        except Exception as e:
            logger.error(f"Error handling job completion for {job.job_id}: {e}")


# Initialize FastAPI app
app = FastAPI(title="CUDA Evaluation Server", version="1.0.0")

# Global device manager (will be initialized on startup)
device_manager: Optional[CUDADeviceManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the CUDA device manager on startup"""
    global device_manager

    # Get CUDA devices from environment variable or use defaults
    cuda_devices_env = os.getenv("CUDA_DEVICES")
    if cuda_devices_env:
        cuda_devices = [device.strip() for device in cuda_devices_env.split(",")]
    else:
        cuda_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # Default

    device_manager = CUDADeviceManager(cuda_devices)
    logger.info("CUDA Evaluation Server started")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "CUDA Evaluation Server is running"}


@app.get("/status")
async def get_status():
    """Get server status and device information"""
    if not device_manager:
        raise HTTPException(status_code=500, detail="Device manager not initialized")

    return {
        "server_status": "running",
        "device_status": device_manager.get_device_status(),
        "queue_info": device_manager.get_queue_info()
    }


@app.post("/evaluate", response_model=Dict[str, str])
async def submit_evaluation(request: EvaluationRequest):
    """Submit a new evaluation job"""
    if not device_manager:
        raise HTTPException(status_code=500, detail="Device manager not initialized")

    try:
        job_id = device_manager.submit_job(request)
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        logger.error(f"Failed to submit evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}", response_model=EvaluationResponse)
async def get_job_status(job_id: str):
    """Get job status and results"""
    if not device_manager:
        raise HTTPException(status_code=500, detail="Device manager not initialized")

    job = device_manager.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Convert KernelExecResult to dict if available
    result_dict = None
    if job.result:
        result_dict = kernel_exec_result_to_dict(job.result)

    return EvaluationResponse(
        job_id=job.job_id,
        status=job.status,
        result=result_dict,
        error=job.error,
        device=job.device,
        queued_at=job.queued_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@app.post("/evaluate_sync")
async def evaluate_sync(request: EvaluationRequest):
    """Submit evaluation and wait for completion (synchronous)"""
    if not device_manager:
        raise HTTPException(status_code=500, detail="Device manager not initialized")

    # Submit job
    job_id = device_manager.submit_job(request)

    # Wait for completion (with timeout)
    timeout = request.timeout or 300
    start_time = time.time()

    while time.time() - start_time < timeout:
        job = device_manager.get_job_status(job_id)
        if job and job.status in ["completed", "failed"]:
            # Convert result to dict
            result_dict = None
            if job.result:
                result_dict = kernel_exec_result_to_dict(job.result)

            return {
                "job_id": job_id,
                "status": job.status,
                "result": result_dict,
                "error": job.error,
                "device": job.device,
                "execution_time": job.completed_at - job.started_at if job.completed_at and job.started_at else None
            }

        await asyncio.sleep(0.5)  # Check every 500ms

    # Timeout
    raise HTTPException(status_code=408, detail="Evaluation timed out")


@app.get("/jobs")
async def list_jobs():
    """List all jobs (active and completed)"""
    if not device_manager:
        raise HTTPException(status_code=500, detail="Device manager not initialized")

    active_jobs = []
    for job_id, job in device_manager.active_jobs.items():
        active_jobs.append({
            "job_id": job_id,
            "status": job.status,
            "problem_id": job.request.problem_id,
            "sample_id": job.request.sample_id,
            "device": job.device,
            "queued_at": job.queued_at,
            "started_at": job.started_at
        })

    completed_jobs = []
    for job_id, job in list(device_manager.completed_jobs.items())[-10:]:  # Last 10 completed jobs
        completed_jobs.append({
            "job_id": job_id,
            "status": job.status,
            "problem_id": job.request.problem_id,
            "sample_id": job.request.sample_id,
            "device": job.device,
            "queued_at": job.queued_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at
        })

    return {
        "active_jobs": active_jobs,
        "completed_jobs": completed_jobs
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start CUDA Evaluation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=6000, help="Port to bind to (default: 6000)")
    parser.add_argument("--cuda-devices", nargs="+",
                        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                        help="List of CUDA devices to use (default: cuda:0 cuda:1 cuda:2 cuda:3)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug"],
                        help="Log level (default: info)")

    args = parser.parse_args()

    # Set environment variable for CUDA devices (will be read by server)
    os.environ["CUDA_DEVICES"] = ",".join(args.cuda_devices)

    print(f"🚀 Starting CUDA Evaluation Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   CUDA Devices: {args.cuda_devices}")
    print(f"   Workers: {args.workers}")
    print(f"   Log Level: {args.log_level}")
    print(f"   Auto-reload: {args.reload}")
    print()

    # Start the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )