CFLAGS  = -O2 -march=native -std=c99 -Wall -Wextra -pedantic
LDFLAGS = -lm

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
	SHARED_EXT = dylib
	SHARED_FLAGS = -shared
	ACCEL_FLAGS = -framework Accelerate
else
	SHARED_EXT = so
	SHARED_FLAGS = -shared -fPIC
	ACCEL_FLAGS =
endif

LIBS = stat_distributions signal_processing matrix_ops engine_kernels semipar_kernels quant_ggml

SHARED_TARGETS = $(addsuffix .$(SHARED_EXT),$(LIBS))

.PHONY: all clean test shared

all: shared test_stat

shared: $(SHARED_TARGETS)

stat_distributions.$(SHARED_EXT): stat_distributions.c stat_distributions.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ stat_distributions.c $(LDFLAGS)

signal_processing.$(SHARED_EXT): signal_processing.c signal_processing.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ signal_processing.c $(LDFLAGS)

matrix_ops.$(SHARED_EXT): matrix_ops.c matrix_ops.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ matrix_ops.c $(LDFLAGS)

engine_kernels.$(SHARED_EXT): engine_kernels.c engine_kernels.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ engine_kernels.c $(LDFLAGS) $(ACCEL_FLAGS)

semipar_kernels.$(SHARED_EXT): semipar_kernels.c semipar_kernels.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ semipar_kernels.c $(LDFLAGS) $(ACCEL_FLAGS)

quant_ggml.$(SHARED_EXT): quant_ggml.c quant_ggml.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ quant_ggml.c $(LDFLAGS)

test_stat: test_stat.c stat_distributions.c signal_processing.c matrix_ops.c
	$(CC) $(CFLAGS) -o $@ test_stat.c stat_distributions.c signal_processing.c matrix_ops.c $(LDFLAGS)

test: test_stat
	./test_stat

clean:
	rm -f *.$(SHARED_EXT) test_stat *.o
