# -------------------------------------------------------------------
# Auto-detect Compiler
# -------------------------------------------------------------------

UNAME_S := $(shell uname -s)

CC = gcc

# 3. For macOS(Darwin)
ifeq ($(UNAME_S),Darwin)
    ifneq ($(shell which gcc-15 2>/dev/null),)
        CC = gcc-15
    else ifneq ($(shell which gcc-14 2>/dev/null),)
        CC = gcc-14
    else ifneq ($(shell which gcc-13 2>/dev/null),)
        CC = gcc-13
    else ifneq ($(shell which gcc-12 2>/dev/null),)
        CC = gcc-12
    endif
endif

# -------------------------------------------------------------------
# Flags & Targets
# -------------------------------------------------------------------

CFLAGS  = -O3 -Wall -std=c11 -fPIC -fopenmp -march=native

# Math library
LDLIBS  = -lm
LDFLAGS = -shared

TARGET  = ../python/libpacman.so
SRC     = $(wildcard *.c)
OBJ     = $(SRC:.c=.o)

BENCH_SRC = benchmark.c
# -------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------

all: info $(TARGET)

info:
	@echo "-------------------------------------------------"
	@echo "System: $(UNAME_S)"
	@echo "Compiler detected: $(CC)"
	@echo "-------------------------------------------------"

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

bench: benchmark.c step_env_apply_level2.c step_apply_sequential.c
	$(CC) $(CFLAGS) -o benchmark benchmark.c step_env_apply_level2.c step_apply_sequential.c $(LDLIBS)

benchmark_bin: $(BENCH_SRC) $(LIB_OBJ)
	$(CC) $(CFLAGS) -o benchmark_c $(BENCH_SRC) $(LIB_OBJ) $(LDLIBS)