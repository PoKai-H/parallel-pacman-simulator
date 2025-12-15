# -------------------------------------------------------------------
# Auto-detect Compiler
# -------------------------------------------------------------------

# 1. 取得作業系統名稱 (Linux 或 Darwin)
UNAME_S := $(shell uname -s)

# 2. 預設編譯器 (適用於 Linux/School Machines)
CC = gcc

# 3. 針對 macOS (Darwin) 的特殊處理
ifeq ($(UNAME_S),Darwin)
    # 嘗試尋找 Homebrew 安裝的 GCC 版本 (從 15 找回 12)
    # shell which 會回傳路徑，如果找不到則為空
    ifneq ($(shell which gcc-15 2>/dev/null),)
        CC = gcc-15
    else ifneq ($(shell which gcc-14 2>/dev/null),)
        CC = gcc-14
    else ifneq ($(shell which gcc-13 2>/dev/null),)
        CC = gcc-13
    else ifneq ($(shell which gcc-12 2>/dev/null),)
        CC = gcc-12
    endif
    # 如果都找不到，維持預設的 gcc (即 Clang)，這可能會導致 OpenMP 錯誤
endif

# -------------------------------------------------------------------
# Flags & Targets
# -------------------------------------------------------------------

# -O3: 最高最佳化
# -march=native: 針對當前 CPU 優化 (注意：在不同機器間傳 .o 檔可能會 crash，這也是為什麼要用 Makefile 重編)
# -fopenmp: 平行化支援
# -fPIC: Shared library 必要
CFLAGS  = -O3 -Wall -std=c11 -fPIC -fopenmp -march=native

# 連結 Math library
LDLIBS  = -lm
LDFLAGS = -shared

TARGET  = ../python/libpacman.so
SRC     = $(wildcard *.c)
OBJ     = $(SRC:.c=.o)

# -------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------

all: info $(TARGET)

# 加一個 info rule 顯示當前使用的編譯器，方便除錯
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