# test.py
import sys
sys.path.insert(0, "build/python")  # where CMake drops the .so

import myapp

app = myapp.TestApp.Create()
print("Successfully created:", app)
