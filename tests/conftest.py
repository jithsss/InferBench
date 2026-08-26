import sys
from pathlib import Path

# Ensure the root of the project is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtimes.environment import configure_nvidia_runtime

# Configure the NVIDIA runtime environment (TensorRT DLL paths, etc.)
# before pytest collects any tests that might import tensorrt at the module level.
def pytest_configure(config):
    configure_nvidia_runtime()
