#!/usr/bin/env python3
"""
Docker entrypoint script for Newsgator application.

This script handles running the Newsgator app in Docker, with options
to generate content, serve content, or both.
"""

import argparse
import logging
import sys
import time
import threading

from rich.console import Console
from rich.logging import RichHandler

from newsgator import Newsgator
from newsgator.web_server import run_server
from newsgator.config import DOCS_DIR

# Initialize Rich console
console = Console()

# Setup logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, console=console, show_time=True, show_path=False)
    ]
)

logger = logging.getLogger(__name__)

def run_content_generation(interval=None):
    """
    Run the content generation process.
    
    Args:
        interval: If set, run content generation periodically at this interval (in hours).
                 If None, run once and exit.
    """
    newsgator = Newsgator()
    
    if interval is None:
        # Run once
        logger.info("Running content generation once")
        newsgator.run()
        return
    
    # Run periodically
    interval_seconds = interval * 3600  # convert hours to seconds
    logger.info(f"Running content generation every {interval} hours")
    
    while True:
        try:
            newsgator.run()
            logger.info(f"Next content generation in {interval} hours")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Content generation stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in content generation: {str(e)}")
            # Wait a bit before retrying
            time.sleep(60)

def main():
    """Main entry point for the Docker container."""
    parser = argparse.ArgumentParser(description="Newsgator Docker Container")
    parser.add_argument(
        "--mode", 
        choices=["generate", "serve", "both"],
        default="both",
        help="Operation mode: generate content, serve content, or both"
    )
    parser.add_argument(
        "--interval", 
        type=float,
        default=None,
        help="Interval (in hours) to periodically generate content. If not set, generate once."
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        # Just generate content
        run_content_generation(args.interval)
    
    elif args.mode == "serve":
        # Just serve existing content
        run_server()
    
    elif args.mode == "both":
        # Generate content and serve it
        if args.interval:
            # Run content generation in a separate thread
            generator_thread = threading.Thread(
                target=run_content_generation,
                args=(args.interval,),
                daemon=True
            )
            generator_thread.start()
        else:
            # Run content generation once
            newsgator = Newsgator()
            newsgator.run()
        
        # Run the web server in the main thread
        run_server()

if __name__ == "__main__":
    main()