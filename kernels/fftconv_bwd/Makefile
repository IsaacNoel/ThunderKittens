GPU := H100

SRC := fftconv_bwd_pc.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

include ../common.mk
