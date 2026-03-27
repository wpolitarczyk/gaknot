import sys
import os
import logging
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import register_line_magic

# 1. Setup Path (Relative to this file)
# Since this file is in gaknot/, the module_path is its parent
lib_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.dirname(lib_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

# Configure Logging IMMEDIATELY so we see logs from imports
def setup_logging(level=logging.INFO):
    """
    Configures logging for the notebook.
    Change level to logging.DEBUG to see more detailed path resolution 
    and preparse information from the gaknot package.
    """
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout,
        force=True
    )

setup_logging()

# 2. Configure IPython Environment
ip = get_ipython()

if ip:
    # Matplotlib inline
    ip.run_line_magic('matplotlib', 'inline')
    
    # Load extensions
    try:
        ip.run_line_magic('load_ext', 'pycodestyle_magic')
    except ImportError:
        pass # Handle case where extension isn't installed

    try:
        ip.run_line_magic('load_ext', 'ipytest')
    except Exception:
        pass # ipytest might not be installable as an extension in all versions

    # Display full output
    InteractiveShell.ast_node_interactivity = 'all'

# 3. Define and Register Custom Magic
try:
    from .utility import import_sage
except ImportError:
    # If the user has already manipulated sys.path, or if this is imported as a package
    try:
        from gaknot.utility import import_sage
    except ImportError:
        print("Warning: Could not import 'gaknot.utility'. Check your path.")
        import_sage = None

@register_line_magic
def preparse(line):
    """
    Custom magic to preparse a sage file using the gaknot utility logic.
    Usage: %preparse signature
    """
    if import_sage is None:
        print("Error: import_sage function not available.")
        return

    package_name = 'gaknot'
    
    # If the user is trying to preparse the main package itself (gaknot.sage)
    if line.strip() == package_name:
        module_to_preparse = package_name
        actual_package = None
        # The file gaknot.sage is INSIDE the gaknot/ directory
        actual_path = os.path.join(module_path, package_name)
    else:
        module_to_preparse = line.strip()
        actual_package = package_name
        actual_path = module_path

    try:
        import_sage(module_to_preparse, package=actual_package, path=actual_path)
        print(f"Successfully preparsed and reloaded: {line}")
    except Exception as e:
        print(f"Error during preparse: {e}")

print("Notebook setup complete. Environment configured.")
