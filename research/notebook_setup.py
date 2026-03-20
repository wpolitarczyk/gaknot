import sys
import os
import logging
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import register_line_magic

# 1. Setup Path (using your existing logic)
# Ensure the parent directory is in sys.path so we can find 'gaknot'
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

    # Display full output
    InteractiveShell.ast_node_interactivity = 'all'

# 3. Define and Register Custom Magic
# We need to import the utility function here. 
# Since we updated sys.path above, this import should work.
try:
    from gaknot.utility import import_sage
except ImportError:
    # Fallback or error handling if gaknot isn't found yet
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

    # Calculate path relative to where this script is located
    # Assuming structure: /project/notebooks/notebook_setup.py
    #                     /project/gaknot/
    
    # You might need to adjust this depending on exactly where 'gaknot' lives relative to this file
    # If notebook_setup.py is in the root, package_path is just 'gaknot'
    # If it's in a subfolder, it might be '../gaknot'
    package_name = 'gaknot'
    package_dir = os.path.join(module_path, package_name)

    try:
        # Note: Your original code passed 'path' as the directory containing the package
        import_sage(line.strip(), package=package_name, path=module_path)
        print(f"Successfully preparsed and reloaded: {line}")
    except Exception as e:
        print(f"Error during preparse: {e}")

# 4. Configure Logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout,
        force=True
    )

# Run default logging setup
setup_logging()

print("Notebook setup complete. Environment configured.")
