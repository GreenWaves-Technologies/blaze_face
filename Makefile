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

io?=host

QUANT_BITS=8
BUILD_DIR=BUILD

#PMSIS_OS?=pulpos

$(info Building GAP8 mode with $(QUANT_BITS) bit quantization)

include model_decl.mk
TRAINED_MODEL=model/face_detection_front.tflite

pulpChip = GAP
PULP_APP = face_detection_front
USE_PMSIS_BSP=1

APP = face_detection_front
APP_SRCS += $(MAIN) post_process.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -gdwarf-2 -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(GAP_SDK_HOME)/utils/power_meas_utils -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DFREQ_CL=$(FREQ_CL) -DFREQ_FC=$(FREQ_FC) -DFREQ_PE=$(FREQ_PE)
ifneq '$(platform)' 'gvsoc'
ifdef GPIO_MEAS
APP_CFLAGS += -DGPIO_MEAS
endif
VOLTAGE?=800
ifeq '$(PMSIS_OS)' 'pulpos'
  APP_CFLAGS += -DVOLTAGE=$(VOLTAGE)
endif
endif

APP_LDFLAGS += -lm

READFS_FILES=$(abspath $(MODEL_TENSORS))

# build depends on the model
build:: model

clean:: clean_model

include model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

