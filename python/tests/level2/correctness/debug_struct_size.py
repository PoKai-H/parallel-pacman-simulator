import sys
import os
import ctypes as C


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import EnvState, lib

def check_size():
    # 1. size of Python ctypes 
    py_size = C.sizeof(EnvState)
    
    lib.get_c_struct_size.restype = C.c_int
    c_size = lib.get_c_struct_size()
    
    print(f"=== Struct Size Diagnostic ===")
    print(f"Python sizeof(EnvState): {py_size} bytes")
    print(f"C      sizeof(EnvState): {c_size} bytes")
    
    if py_size == c_size:
        print("MATCH! Sizes are consistent.")
    else:
        print(f"MISMATCH! Difference: {abs(py_size - c_size)} bytes.")
        print("Hint: If Python is smaller, you are missing fields in pacman_env.py.")
        print("Hint: If diff is 4/8 bytes, check types (int vs long) or Padding.")

if __name__ == "__main__":
    check_size()