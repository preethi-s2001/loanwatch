"""
Executes cells in notebooks/01_EDA.ipynb and embeds text outputs and base64 plots.
"""

import base64
import io
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

NOTEBOOK_PATH = Path("notebooks/01_EDA.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution namespace
exec_globals = {"__name__": "__main__"}

# Intercept plt.show so figures are not closed before capturing
original_show = plt.show
captured_figures = []

def custom_show(*args, **kwargs):
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)
        captured_figures.append(fig)

plt.show = custom_show

for cell_idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_str = "".join(cell["source"])
        code_str_adj = code_str.replace("os.path.join('..', 'data'", "os.path.join('data'")

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        captured_figures.clear()
        plt.close("all")

        try:
            exec(code_str_adj, exec_globals)
            stdout_text = stdout_capture.getvalue()
        except Exception as e:
            stdout_text = f"Execution Error: {e}"
            print(f"Error in cell {cell_idx}: {e}")
        finally:
            sys.stdout = old_stdout

        outputs = []
        if stdout_text.strip():
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True)
            })

        # Process any figures that were shown or created
        all_figs = captured_figures if captured_figures else [plt.figure(i) for i in plt.get_fignums()]
        for fig in all_figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {}
            })

        plt.close("all")
        cell["outputs"] = outputs
        cell["execution_count"] = cell_idx

plt.show = original_show

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully executed and populated outputs in all cells of {NOTEBOOK_PATH}!")
