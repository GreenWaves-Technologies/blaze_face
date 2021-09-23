# Copyright 2021 GreenWaves Technologies, SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif
MODEL_PREFIX = face_detection_front
include common.mk

#IMAGE=$(CURDIR)/images/croppedpgmfile1.ppm
IMAGE=$(CURDIR)/images/francesco_cropped_r.ppm

io=stdout

QUANT_BITS=8
BUILD_DIR=BUILD

PMSIS_OS?=pulpos

$(info Building GAP8 mode with $(QUANT_BITS) bit quantization)

ifeq ($(MODEL_FP16), 1)
	NNTOOL_SCRIPT=model/nntool_script_fp16
	MODEL_SUFFIX = _FP16
	MAIN = main_fp16.c
	APP_CFLAGS += -DFLOAT_POST_PROCESS -DSTD_FLOAT
	CLUSTER_STACK_SIZE=6048
else
	MODEL_SQ8=1
	NNTOOL_SCRIPT=model/nntool_script
	MODEL_SUFFIX = _SQ8BIT
	MAIN = main.c
endif

#LOAD A TFLITE QUANTIZED GRAPH
NNTOOL_EXTRA_FLAGS=

include model_decl.mk
TRAINED_MODEL=model/face_detection_front.tflite

# Here we set the memory allocation for the generated kernels
# REMEMBER THAT THE L1 MEMORY ALLOCATION MUST INCLUDE SPACE
# FOR ALLOCATED STACKS!
CLUSTER_STACK_SIZE?=4028
CLUSTER_SLAVE_STACK_SIZE=1024
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)

MODEL_L1_MEMORY=$(shell expr $(TARGET_L1_MEMORY) \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY=$(TARGET_L2_MEMORY)
MODEL_L3_MEMORY=$(TARGET_L3_MEMORY)

# hram - HyperBus RAM
# qspiram - Quad SPI RAM
MODEL_L3_EXEC=hram
# hflash - HyperBus Flash
# qpsiflash - Quad SPI Flash
MODEL_L3_CONST=hflash

pulpChip = GAP
PULP_APP = face_detection_front
USE_PMSIS_BSP=1

APP = face_detection_front
APP_SRCS += $(MAIN) post_process.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE)

#APP_CFLAGS += -DPERF

APP_LDFLAGS += -lm

READFS_FILES=$(abspath $(MODEL_TENSORS))

# all depends on the model
all:: model

clean:: clean_model

include model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

