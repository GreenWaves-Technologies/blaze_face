# Copyright (C) 2021 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

# The training of the model is slightly different depending on
# the quantization. This is because in 8 bit mode we used signed
# 8 bit so the input to the model needs to be shifted 1 bit

MODEL_SUFFIX?=

MODEL_PREFIX?=GapFlow

MODEL_PYTHON=python3

# Increase this to improve accuracy
TRAINING_EPOCHS?=1
MODEL_COMMON ?= common
MODEL_COMMON_INC ?= $(GAP_SDK_HOME)/libs/gap_lib/include
MODEL_COMMON_SRC ?= $(GAP_SDK_HOME)/libs/gap_lib/img_io
MODEL_COMMON_SRC_FILES ?= ImgIO.c
MODEL_COMMON_SRCS = $(realpath $(addprefix $(MODEL_COMMON_SRC)/,$(MODEL_COMMON_SRC_FILES)))
MODEL_HEADERS = $(MODEL_COMMON)/headers
MODEL_HEADER = $(MODEL_PREFIX)Info.h
MODEL_TRAIN = model/train.py
MODEL_BUILD = BUILD_MODEL$(MODEL_SUFFIX)
MODEL_TRAIN_BUILD = BUILD_TRAIN$(TRAIN_SUFFIX)
MODEL_H5 = $(MODEL_TRAIN_BUILD)/$(MODEL_PREFIX).h5

MODEL_PATH = $(MODEL_BUILD)/$(MODEL_PREFIX).tflite

TENSORS_DIR = $(MODEL_BUILD)/tensors
MODEL_TENSORS = $(MODEL_BUILD)/$(MODEL_PREFIX)_L3_Flash_Const.dat

MODEL_EXPRESSIONS = $(MODEL_BUILD)/Expression_Kernels.c
MODEL_STATE = $(MODEL_BUILD)/$(MODEL_PREFIX).json
MODEL_SRC = $(MODEL_PREFIX)Model.c
MODEL_GEN = $(MODEL_BUILD)/$(MODEL_PREFIX)Kernels 
MODEL_GEN_C = $(addsuffix .c, $(MODEL_GEN))
MODEL_GEN_CLEAN = $(MODEL_GEN_C) $(addsuffix .h, $(MODEL_GEN))
MODEL_GEN_EXE = $(MODEL_BUILD)/GenTile

MODEL_GENFLAGS_EXTRA =

EXTRA_GENERATOR_SRC =

IMAGES = images
RM=rm -f

CLUSTER_STACK_SIZE?=4096
CLUSTER_SLAVE_STACK_SIZE ?= 512
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 8)
MODEL_L1_MEMORY=$(shell expr $(TARGET_L1_SIZE) \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY=$(TARGET_L2_SIZE)
MODEL_L3_MEMORY=$(TARGET_L3_SIZE)

NNTOOL=nntool
include $(RULES_DIR)/at_common_decl.mk
$(info GEN ... $(CNN_GEN))

