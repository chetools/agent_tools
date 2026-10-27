# Verification entry point (AGENTS.md section 3, step 6).
#   make verify        every check below; fails on the first failing group
#   make verify-python Python reference implementations against independent methods (pytest)
#   make verify-gpu    stored reference data reproduced, then WGSL kernels against it in headless Chromium
PYTHON ?= python3

.PHONY: verify verify-python verify-gpu

verify: verify-python verify-gpu

verify-python:
	$(PYTHON) -m pytest -q library/tests/python

verify-gpu:
	$(PYTHON) library/tests/gpu/gen_reference.py --check
	node library/tests/gpu/run.mjs
