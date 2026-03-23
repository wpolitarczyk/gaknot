import sys
import os
import logging
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import register_line_magic

# Configure Logging IMMEDIATELY so we see logs from imports
def setup_logging(level=logging.INFO):
    """
    Configures logging for the notebook.
    Change level to logging.DEBUG to see more detailed path resolution 
    and preparse information from the gaknot_lib package.
    """
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout,
        force=True
    )

setup_logging()

# 1. Setup Path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

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
    from gaknot_lib.utility import import_sage
except ImportError:
    print("Warning: Could not import 'gaknot_lib.utility'. Check your path.")
    import_sage = None

@register_line_magic
def preparse(line):
    """
    Custom magic to preparse a sage file using the gaknot_lib utility logic.
    Usage: %preparse signature
    """
    if import_sage is None:
        print("Error: import_sage function not available.")
        return

    package_name = 'gaknot_lib'
    module_to_preparse = line.strip()

    # Special handling for the main package entry point (gaknot.sage)
    if module_to_preparse == package_name or module_to_preparse == 'gaknot':
        module_to_preparse = 'gaknot'
    
    # All files inside gaknot_lib should be imported with package='gaknot_lib' 
    # and path set to the project root.
    actual_package = package_name
    actual_path = module_path

    try:
        import_sage(module_to_preparse, package=actual_package, path=actual_path)
        print(f"Successfully preparsed and reloaded: {module_to_preparse}")
    except Exception as e:
        print(f"Error during preparse: {e}")

print("Notebook setup complete. Environment configured.")
