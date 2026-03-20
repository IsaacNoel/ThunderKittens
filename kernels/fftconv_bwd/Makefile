GPU := H100

# Two-kernel version (default)
SRC := fftconv_bwd_pc.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Fused kernel (single kernel, du + dk_f together)
# SRC := fftconv_bwd_fused.cu
# OUT := _C_fused$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# CMD := python test_correctness_fused.py
# NVCCFLAGS := -DTORCH_COMPILE
# CONFIG := pytorch

include ../common.mk
