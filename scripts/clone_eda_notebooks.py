"""Clone the Brunswick EDA notebook for halfmile, lalor, and sudbury."""
import json
import os

# Determine paths relative to script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "notebooks")

src = os.path.join(NB_DIR, "01_eda_brunswick.ipynb")
with open(src, "r") as f:
    nb = json.load(f)

for asset in ["halfmile", "lalor", "sudbury"]:
    nb_copy = json.loads(json.dumps(nb))  # deep copy

    # Fix markdown cell title and description
    nb_copy["cells"][0]["source"] = [
        s.replace("Brunswick", asset.capitalize()).replace("brunswick", asset)
        for s in nb_copy["cells"][0]["source"]
    ]

    # Fix ASSET_NAME in code cells
    for cell in nb_copy["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                if line.startswith('ASSET_NAME = '):
                    new_source.append(f'ASSET_NAME = "{asset}"\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source

    out_path = os.path.join(NB_DIR, f"01_eda_{asset}.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb_copy, f, indent=2)
    print(f"Created: {out_path}")

print("Done — all 3 cloned notebooks created.")
