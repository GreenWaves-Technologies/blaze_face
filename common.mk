# Copyright (C) 2020 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

AT_INPUT_WIDTH=128
AT_INPUT_HEIGHT=128
AT_INPUT_COLORS=3

ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
	FREQ_CL?=370
	FREQ_FC?=370
	TARGET_L1_MEMORY=128000
	TARGET_L2_MEMORY=1450000
	TARGET_L3_MEMORY=8000000
else
	ifeq '$(TARGET_CHIP)' 'GAP8_V3'
		FREQ_CL?=175
	else
		FREQ_CL?=50
	endif
	FREQ_FC?=250
	TARGET_L1_MEMORY=64000
	TARGET_L2_MEMORY=250000
	TARGET_L3_MEMORY=8000000
endif

ifeq ($(MODEL_FP16), 1)
	NNTOOL_SCRIPT=model/nntool_script_fp16
	MODEL_SUFFIX = _FP16
	MAIN = main_fp16.c
	APP_CFLAGS += -DFLOAT_POST_PROCESS -DSTD_FLOAT
	CLUSTER_STACK_SIZE=6048
else
MODEL_SQ8=1
ifeq ($(MODEL_NE16), 1)
	NNTOOL_SCRIPT=model/nntool_script_ne16
	MODEL_SUFFIX = _NE16
	CLUSTER_STACK_SIZE=6048
else
ifeq ($(MODEL_HWC), 1)
        NNTOOL_SCRIPT=model/nntool_script_hwc
        MODEL_SUFFIX = _HWC
else
	NNTOOL_SCRIPT=model/nntool_script
	MODEL_SUFFIX = _SQ8BIT
endif
endif
endif
