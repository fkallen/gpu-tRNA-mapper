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

# ARCH_SPECIFIER = \
# 	-gencode=arch=compute_70,code=sm_70 \
# 	-gencode=arch=compute_80,code=sm_80 \
# 	-gencode=arch=compute_89,code=sm_89

ARCH_SPECIFIER = -arch=native

NVCC_FLAGS = $(ARCH_SPECIFIER) --threads 3 -lineinfo --extended-lambda -Xcompiler "-fopenmp" $(DEFINES) $(INCLUDES)

COMPILE = nvcc $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@

EXECUTABLE = gpu_trna_mapper

.PHONY:	all
all:	$(EXECUTABLE)

OBJECTS = $(BUILDDIR)/main.o $(BUILDDIR)/sam_helpers.o $(BUILDDIR)/parsing.o $(BUILDDIR)/smallkernels.o \
	$(BUILDDIR)/read_parser_worker.o $(BUILDDIR)/output_writer_worker.o $(BUILDDIR)/cpu_traceback_worker.o \
	$(BUILDDIR)/gpu_topscores_worker_local.o $(BUILDDIR)/gpu_topscores_worker_semiglobal.o


BUILDDIR = build

$(shell mkdir -p $(BUILDDIR))

$(BUILDDIR)/main.o: main.cu 
	$(COMPILE)

$(BUILDDIR)/sam_helpers.o: sam_helpers.cpp 
	$(COMPILE)

$(BUILDDIR)/parsing.o: parsing.cu 
	$(COMPILE)

$(BUILDDIR)/smallkernels.o: smallkernels.cu 
	$(COMPILE)

$(BUILDDIR)/read_parser_worker.o: read_parser_worker.cu 
	$(COMPILE)

$(BUILDDIR)/output_writer_worker.o: output_writer_worker.cu 
	$(COMPILE)

$(BUILDDIR)/cpu_traceback_worker.o: cpu_traceback_worker.cu 
	$(COMPILE)

$(BUILDDIR)/gpu_topscores_worker_local.o: gpu_topscores_worker_local.cu 
	$(COMPILE)

$(BUILDDIR)/gpu_topscores_worker_semiglobal.o: gpu_topscores_worker_semiglobal.cu 
	$(COMPILE)



$(EXECUTABLE): $(OBJECTS)
	nvcc $^ -o $(EXECUTABLE) $(NVCC_FLAGS) $(LDFLAGS)



clean:
	rm -rf build/* $(EXECUTABLE)
