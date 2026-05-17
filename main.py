#!/usr/bin/env python3
"""
RAGking v2.1 — RAG Evaluation & Ranking Framework
by Lander Liquicota · Thesis Project 2025

Entry point — run this file to start the CLI.
"""
import sys
import os

# Ensure the project root is always on sys.path regardless of
# where the user runs the script from.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.menu import main

if __name__ == "__main__":
    main()