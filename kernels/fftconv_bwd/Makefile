GPU    := H100
CONFIG := pytorch
NVCCFLAGS := -DTORCH_COMPILE

PY_EXT := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT    := _C$(PY_EXT)

SRC := fftconv_bwd_pc.cu
CMD := python test_correctness.py

include ../common.mk

# ── Additional targets ────────────────────────────────────────────────────────

# Run tests (build first if needed)
test: $(OUT)
	python test_correctness.py

# Benchmarks (require GPU)
benchmark: $(OUT)
	python benchmarks/run_benchmarks.py

.PHONY: test benchmark
