# Configurable parameters
GPU_COMPILE_THREADS ?= 1
TARGET_GPU_ARCH ?= native
INSTALL_PREFIX ?= /usr/local
PARASAIL_INCLUDE_DIR ?= parasail-2.6.2
PARASAIL_LIB_DIR ?= parasail-2.6.2/build

# Not for configuration
RMM_INCDIR = rmm-24.06.00/include
NVTX_INCDIR = NVTX/c/include
PARASAIL_DIR = parasail-2.6.2
GPU_ALIGNMENT_API = gpu_api/include

DIALECT = -std=c++17
OPTIMIZATION = -O3 -g
WARNINGS = -Xcompiler="-Wall -Wextra"
BUILDDIR = build

DEFINES = -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DNDEBUG
INCLUDES = -I$(RMM_INCDIR) -I$(NVTX_INCDIR) -I$(PARASAIL_INCLUDE_DIR) -I$(GPU_ALIGNMENT_API)
LDFLAGS = -lz -ldl -L $(PARASAIL_LIB_DIR) -l parasail -Xcompiler \"-Wl,-rpath,$(PARASAIL_LIB_DIR)\"

NVCC_FLAGS = -arch=$(TARGET_GPU_ARCH) --threads $(GPU_COMPILE_THREADS) -lineinfo --extended-lambda -Xcompiler "-fopenmp" $(DEFINES) $(INCLUDES)

COMPILE = nvcc $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@
COMPILE_ECHO = echo "Compiling $@"; $(COMPILE)

EXECUTABLE = gpu-tRNA-mapper

.PHONY:	all
all:	$(EXECUTABLE)

CU_OBJECTS = $(addprefix $(BUILDDIR)/, \
    main.o parsing.o smallkernels.o read_parser_worker.o output_writer_worker.o \
    cpu_traceback_worker.o gpu_topscores_worker_local.o gpu_topscores_worker_semiglobal.o \
)

CPP_OBJECTS = $(BUILDDIR)/sam_helpers.o

$(shell mkdir -p $(BUILDDIR))

$(CU_OBJECTS): $(BUILDDIR)/%.o: %.cu
	@ $(COMPILE_ECHO)

$(CPP_OBJECTS): $(BUILDDIR)/%.o: %.cpp
	@ $(COMPILE_ECHO)

$(EXECUTABLE): $(CU_OBJECTS) $(CPP_OBJECTS)
	@ echo "Linking $(EXECUTABLE)"
	@ nvcc $^ -o $(EXECUTABLE) $(NVCC_FLAGS) $(LDFLAGS)

.PHONY:	clean
clean:
	rm -rf build/* $(EXECUTABLE)

.PHONY:	install
install: $(EXECUTABLE)
	install -D $(EXECUTABLE) "$(INSTALL_PREFIX)/bin"

