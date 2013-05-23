EXT := cu 

OBJS := gprepel.o

#compiler/preprocessor options
INCS := -I/opt/cuda/include -arch sm_11 --pre-include gcc_cuda_compat.h
PARAMS := -Xcompiler "-I/usr/lib64/R/include -fpic"

#linker options
LD_PARAMS := -Xlinker "-L/usr/lib64/R/lib -lR ,-rpath,/opt/cuda/lib64"
LIBS :=  -L/opt/cuda/lib64 -lcublas -lcuda

TARGETS := gpRepel.so

CC := /opt/cuda/bin/nvcc -D _gpRepel_ 

all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(CC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(CC) -c $(INCS) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
