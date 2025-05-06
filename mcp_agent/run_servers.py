#!/usr/bin/env python3
"""
MCP Servers Orchestration Script
Starts and manages all MCP servers for the multi-modal agent
"""

import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

servers = [
    {"script": "servers/vision_server.py", "name": "Vision Model Server", "port": 8000},
    {"script": "servers/llm_server.py", "name": "LLM Server", "port": 8001},
    {"script": "servers/search_server.py", "name": "Search Server", "port": 8002},
]

processes = []


def start_servers():
    """Start all MCP servers"""
    for server in servers:
        logger.info(f"Starting {server['name']} on port {server['port']}...")
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), server["script"]
        )
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(
            {"process": process, "name": server["name"], "port": server["port"]}
        )
        logger.info(f"{server['name']} started with PID {process.pid}")
        time.sleep(2)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(
                f"Failed to start {server['name']}. Exit code: {process.returncode}"
            )
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
        else:
            logger.info(f"{server['name']} is running")


def check_server_status():
    """Check the status of all running server processes"""
    for proc_info in processes:
        process = proc_info["process"]
        name = proc_info["name"]
        if process.poll() is None:
            logger.info(f"{name} (PID {process.pid}) is running")
        else:
            stdout, stderr = process.communicate()
            logger.warning(
                f"{name} (PID {process.pid}) has terminated. Exit code: {process.returncode}"
            )
            logger.warning(f"STDOUT: {stdout}")
            logger.warning(f"STDERR: {stderr}")


def stop_servers():
    """Stop all running server processes"""
    for proc_info in processes:
        process = proc_info["process"]
        name = proc_info["name"]
        if process.poll() is None:
            logger.info(f"Stopping {name} (PID {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
                logger.info(f"{name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} did not terminate gracefully, forcing...")
                process.kill()
                process.wait()
                logger.info(f"{name} forcefully stopped")
        else:
            logger.info(f"{name} was already stopped")

    logger.info("All servers stopped")


def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully shut down servers"""
    logger.info("\nReceived interrupt signal. Shutting down servers...")
    stop_servers()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        start_servers()
        logger.info("\nAll servers are running. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(30)
            check_server_status()
    except KeyboardInterrupt:
        logger.info("\nShutting down servers...")
        stop_servers()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        stop_servers()
        sys.exit(1)
