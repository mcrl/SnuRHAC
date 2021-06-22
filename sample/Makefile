CPU_COMPILER = g++
CUDA_COMPILER = nvcc

CUDA_INCLUDES = -I$(SNURHACCUDA)/include
CUDA_LIBRARIES = -L$(SNURHACCUDA)/lib64
SNURHAC_LIBRARIES = -L$(SNURHACROOT)/runtime/lib

CUDA_CPU_CFLAGS = $(CUDA_INCLUDES) $(CUDA_LIBRARIES)
CUDA_CPU_LDFLAGS = -lcudart

SNURHAC_CPU_CFLAGS = $(CUDA_INCLUDES) $(SNURHAC_LIBRARIES)
SNURHAC_CPU_LDFLAGS = -lsnurhac

GPU_CODEGEN_FLAGS = -arch=compute_70 -code=sm_70,compute_70

CUDA_GPU_CFLAGS = $(GPU_CODEGEN_FLAGS) -ccbin=$(CPU_COMPILER)
SNURHAC_GPU_CFLAGS = $(GPU_CODEGEN_FLAGS) -ccbin=$(CPU_COMPILER)

SOURCES = syrk.cu

CUDA_TARGET = run.cuda
SNURHAC_TARGET = run.snurhac

all: cuda snurhac

cuda: $(CUDA_TARGET)

snurhac: $(SNURHAC_TARGET)

$(CUDA_TARGET): syrk.cuda.o
	$(CPU_COMPILER) $^ $(CUDA_CPU_CFLAGS) $(CUDA_CPU_LDFLAGS) -o $@

$(SNURHAC_TARGET): syrk.snurhac.o
	$(CPU_COMPILER) $^ $(SNURHAC_CPU_CFLAGS) $(SNURHAC_CPU_LDFLAGS) -o $@

syrk.cuda.o: $(SOURCES)
	$(CUDA_COMPILER) -c $^ $(CUDA_GPU_CFLAGS) -o $@ 

syrk.snurhac.o: $(SOURCES)
	$(CUDA_COMPILER) -c $^ $(SNURHAC_GPU_CFLAGS) -o $@ 

.PHONY: clean run

clean:
	rm -f *.o *.tmp.cu *.ptx *.cubin *.fatbin *.snurhac $(CUDA_TARGET) $(SNURHAC_TARGET)

run: all
	./$(CUDA_TARGET)
	./$(SNURHAC_TARGET)
