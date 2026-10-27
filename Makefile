# `make verify` runs every test in the library. Any failure makes it exit non-zero.
.PHONY: verify
verify:
	cargo test --release --manifest-path library/rust/Cargo.toml -- --nocapture
	python -m pytest -s library/tests/python
