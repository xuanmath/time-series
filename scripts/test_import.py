"""
Test Auto Optimizer
"""

import sys
from pathlib import Path

# Get paths
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")

# Add to path
sys.path.insert(0, str(project_root))

# Try importing
try:
    from scripts import auto_optimizer
    print("Import success!")
    print(dir(auto_optimizer))
except Exception as e:
    print(f"Import error: {e}")
    
# Try direct exec
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "auto_optimizer", 
        project_root / "scripts" / "auto_optimizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Direct import success!")
    print(f"AutoOptimizer: {module.AutoOptimizer}")
except Exception as e:
    print(f"Direct import error: {e}")
