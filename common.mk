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
