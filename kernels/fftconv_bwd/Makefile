GPU    := H100
CONFIG := pytorch
NVCCFLAGS := -DTORCH_COMPILE

PY_EXT := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT    := _C$(PY_EXT)

# Default: build two-kernel version (fftconv_bwd_pc.cu)
SRC := fftconv_bwd_pc.cu
CMD := python test_correctness.py

include ../common.mk

# ── Additional targets ────────────────────────────────────────────────────────

# Build fused kernel (overwrites _C.so — both tests import from _C)
fused:
	$(MAKE) SRC=fftconv_bwd_fused.cu CMD="python test_correctness_fused.py" all

# Run tests (build first if needed)
test: $(OUT)
	python test_correctness.py

test-fused:
	$(MAKE) fused
	python test_correctness_fused.py

.PHONY: fused test test-fused
