import sys
import os
import logging
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

# 1. Setup Path (Relative to this file)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(root_dir, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure Logging
def setup_logging(level=logging.INFO):
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
    ip.run_line_magic('matplotlib', 'inline')
    try:
        ip.run_line_magic('load_ext', 'ipytest')
    except Exception:
        pass
    InteractiveShell.ast_node_interactivity = 'all'

print("Notebook setup complete. Environment configured for standard Python imports.")
