import importlib.util
import os
import sys
import types

# Tests historically import `from tools.quad_partition_validator import ...`.
# Avoid executing tools/__init__.py (which pulls in heavy optional dependencies
# such as gmsh/sklearn) by constructing a minimal fake `tools` package that
# contains only the validator submodule.
_domain_partition_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tools_dir = os.path.join(_domain_partition_dir, "tools")

_tools_pkg = types.ModuleType("tools")
_tools_pkg.__path__ = [_tools_dir]
sys.modules["tools"] = _tools_pkg


def _load_tool(name):
    path = os.path.join(_tools_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"tools.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tools.{name}"] = module
    setattr(_tools_pkg, name, module)
    spec.loader.exec_module(module)
    return module


_load_tool("quad_partition_validator")
