# Makefile for standalone CUDA module compilation

# Compiler and tools
NVCC ?= nvcc
PYTHON ?= .venv/bin/python

# Directories
EXT_NAME ?= _torch_ext
SRC_DIR ?= src/my_proj/ext/$(EXT_NAME)/lib
BUILD_DIR ?= build

# Auto-detect CUDA architecture from nvidia-smi, fallback to sm_75
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
CUDA_ARCH ?= $(if $(DETECTED_ARCH),sm_$(DETECTED_ARCH),sm_75)

# Get PyTorch include paths
TORCH_INCLUDE_DIR := $(shell $(PYTHON) -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))" 2>/dev/null)
TORCH_INCLUDES := $(if $(TORCH_INCLUDE_DIR),-I$(TORCH_INCLUDE_DIR) -I$(TORCH_INCLUDE_DIR)/torch/csrc/api/include,)

# Compiler flags
NVCC_FLAGS ?= -std=c++17 -O2 --expt-relaxed-constexpr
CUDA_FLAGS ?= -arch=$(CUDA_ARCH) -DWITH_CUDA -DSTANDALONE_BUILD
INCLUDES ?= -I$(SRC_DIR) $(TORCH_INCLUDES)

# Source and target
PROGRAM ?= reduce_add
PROGRAM_H := $(SRC_DIR)/$(PROGRAM).h
PROGRAM_CU := $(SRC_DIR)/$(PROGRAM).cu
PROGRAM_BIN := $(BUILD_DIR)/$(PROGRAM).bin

.PHONY: all
all: $(PROGRAM_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PROGRAM_BIN): $(PROGRAM_CU) $(PROGRAM_H) $(SRC_DIR)/utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_FLAGS) $(INCLUDES) -o $@ $<

.PHONY: run
run: $(PROGRAM_BIN)
	$(PROGRAM_BIN)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all     - Build (default)"
	@echo "  run     - Build and run"
	@echo "  clean   - Remove build artifacts"
	@echo ""
	@echo "Variables (can override):"
	@echo "  NVCC        = $(NVCC)"
	@echo "  PYTHON      = $(PYTHON)"
	@echo "  CUDA_ARCH   = $(CUDA_ARCH)"
	@echo "  SRC_DIR     = $(SRC_DIR)"
	@echo "  PROGRAM     = $(PROGRAM)"
