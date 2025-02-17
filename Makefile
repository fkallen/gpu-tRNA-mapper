RMM_INCDIR = rmm-24.06.00/include
NVTX_INCDIR = NVTX/c/include
PARASAIL_DIR = parasail-2.6.2
GPU_ALIGNMENT_API = gpu_api/include


DIALECT = -std=c++17
OPTIMIZATION = -O3 -g
WARNINGS = -Xcompiler="-Wall -Wextra"

DEFINES = -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DNDEBUG
INCLUDES = -I$(RMM_INCDIR) -I$(NVTX_INCDIR) -I$(PARASAIL_DIR) -I$(GPU_ALIGNMENT_API)
LDFLAGS = -L$(PARASAIL_DIR)/build -lz -ldl -lparasail

TARGET_GPU_ARCH = $(GPUARCH)
ifeq ($(TARGET_GPU_ARCH),)
	TARGET_GPU_ARCH = native
endif

GPU_COMPILE_THREADS = $(GPUARCH_NUM_COMPILE_THREADS)
ifeq ($(GPU_COMPILE_THREADS),)
	GPU_COMPILE_THREADS = 1
endif

INSTALL_PREFIX = $(PREFIX)
ifeq ($(INSTALL_PREFIX),)
	INSTALL_PREFIX = /usr/local
endif

NVCC_FLAGS = -arch=$(TARGET_GPU_ARCH) --threads $(GPU_COMPILE_THREADS) -lineinfo --extended-lambda -Xcompiler "-fopenmp" $(DEFINES) $(INCLUDES)

COMPILE = nvcc $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@
COMPILE_ECHO = echo "Compiling $@"; $(COMPILE)

EXECUTABLE = gpu-tRNA-mapper

.PHONY:	all
all:	$(EXECUTABLE)

OBJECTS = $(BUILDDIR)/main.o $(BUILDDIR)/sam_helpers.o $(BUILDDIR)/parsing.o $(BUILDDIR)/smallkernels.o \
	$(BUILDDIR)/read_parser_worker.o $(BUILDDIR)/output_writer_worker.o $(BUILDDIR)/cpu_traceback_worker.o \
	$(BUILDDIR)/gpu_topscores_worker_local.o $(BUILDDIR)/gpu_topscores_worker_semiglobal.o


BUILDDIR = build

$(shell mkdir -p $(BUILDDIR))

$(BUILDDIR)/main.o: main.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/sam_helpers.o: sam_helpers.cpp 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/parsing.o: parsing.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/smallkernels.o: smallkernels.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/read_parser_worker.o: read_parser_worker.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/output_writer_worker.o: output_writer_worker.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/cpu_traceback_worker.o: cpu_traceback_worker.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/gpu_topscores_worker_local.o: gpu_topscores_worker_local.cu 
	@ $(COMPILE_ECHO)

$(BUILDDIR)/gpu_topscores_worker_semiglobal.o: gpu_topscores_worker_semiglobal.cu 
	@ $(COMPILE_ECHO)



$(EXECUTABLE): $(OBJECTS)
	@ echo "Linking $(EXECUTABLE)"
	@ nvcc $^ -o $(EXECUTABLE) $(NVCC_FLAGS) $(LDFLAGS)


.PHONY:	clean
clean:
	rm -rf build/* $(EXECUTABLE)

.PHONY:	install
install: $(EXECUTABLE)
	mkdir -p "$(INSTALL_PREFIX)/bin"
	cp $(EXECUTABLE) "$(INSTALL_PREFIX)/bin"

