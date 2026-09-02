#!/usr/bin/env bash
set -euo pipefail
python -m py_compile qwen35_t4_endpoint.py
bash -n benchmark_endpoint.sh .auto/measure.sh
