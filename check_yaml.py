import sys
print(sys.executable)  # Shows which Python interpreter is being used
print(sys.path)        # Shows the module search paths

try:
    import yaml
    print("PyYAML is installed and working!")
    print(yaml.__file__)  # Shows the location of the yaml module
except ImportError:
    print("PyYAML is not accessible from this Python environment")