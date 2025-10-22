#!/usr/bin/env python3
"""
Submit Axolotl fine-tuning job to RunPod API
"""

import os
import time
import requests
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load Axolotl configuration from YAML file.

    Args:
        config_path: Path to config YAML file (REQUIRED)

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


class RunPodAxolotlClient:
    """Client for submitting Axolotl fine-tuning jobs to RunPod"""
    
    def __init__(self, api_key: str, endpoint_id: str = "aid5ds15550pph"):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    
    def submit_training_job(
        self,
        config_path: str
    ) -> Dict[str, Any]:
        """
        Submit an Axolotl training job to RunPod

        Args:
            config_path: Path to YAML config file

        Returns:
            Response dict with job ID

        Examples:
            # Use default config file
            >>> client.submit_training_job('finetuning/axolotl/configs/default_config.yaml')

            # Use custom config file
            >>> client.submit_training_job('finetuning/axolotl/configs/my_custom_config.yaml')

        Note:
            - Make a copy of the config file and modify it for your needs
            - Dataset path in config must be accessible by RunPod (URL or /workspace path)
            - HF_TOKEN and WANDB_API_KEY will be injected from environment variables
        """
        # Load config from file
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)

        # Extract credentials from environment variables for RunPod API structure
        credentials = {}
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            credentials['hf_token'] = hf_token

        wandb_key = os.environ.get('WANDB_API_KEY')
        if wandb_key:
            credentials['wandb_api_key'] = wandb_key

        # Prepare the input payload following RunPod Axolotl serverless structure
        # Try using 'args' instead of 'config' as the key
        input_data = {
            'args': config,
            'config': config  # Include both for compatibility
        }

        # Add credentials if available
        if credentials:
            input_data['credentials'] = credentials

        payload = {'input': input_data}

        # Submit the job
        print(f"\nSubmitting fine-tuning job to RunPod endpoint: {self.endpoint_id}")
        print(f"Base model: {config.get('base_model')}")
        print(f"Epochs: {config.get('num_epochs')}")

        response = requests.post(
            f'{self.base_url}/run',
            headers=self.headers,
            json=payload
        )

        response.raise_for_status()
        result = response.json()

        print(f"\n✓ Job submitted successfully!")
        print(f"Job ID: {result.get('id')}")
        print(f"Status: {result.get('status')}")

        return result
    
    def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a running job"""
        response = requests.get(
            f'{self.base_url}/status/{job_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def stream_logs(self, job_id: str) -> Dict[str, Any]:
        """Get streaming logs for a job"""
        response = requests.get(
            f'{self.base_url}/stream/{job_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job"""
        response = requests.post(
            f'{self.base_url}/cancel/{job_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_output(self, job_id: str) -> Dict[str, Any]:
        """Get the output/results from a completed job"""
        status = self.check_status(job_id)
        if status.get('status') == 'COMPLETED':
            return status.get('output', {})
        else:
            print(f"Job status: {status.get('status')}")
            return {}
    
    def wait_for_completion(self, job_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """
        Wait for job completion and poll status
        
        Args:
            job_id: The job ID to monitor
            check_interval: Seconds between status checks
        """
        print(f"\nMonitoring job {job_id}...")
        
        while True:
            status_response = self.check_status(job_id)
            status = status_response.get('status')
            
            print(f"Status: {status}")
            
            if status == 'COMPLETED':
                print("\n✓ Job completed successfully!")
                return status_response
            elif status in ['FAILED', 'CANCELLED']:
                print(f"\n✗ Job {status.lower()}")
                return status_response
            
            time.sleep(check_interval)