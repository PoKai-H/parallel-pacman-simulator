# csrc/Makefile
CC      = gcc
CFLAGS  = -O2 -std=c11 -fPIC
LDFLAGS = -shared

SRC     = step_apply_sequential.c
OBJ     = $(SRC:.c=.o)

all: ../python/libpacman.so

%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

../python/libpacman.so: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $(OBJ)

clean:
	rm -f $(OBJ) ../python/libpacman.so
