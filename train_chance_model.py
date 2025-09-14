#!/usr/bin/env python3
"""Standalone training script for chance model - estimates probability that a ride starts at given location."""

import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ml.train_chance import main

if __name__ == "__main__":
    main()