"""Make the executed example notebook preview as static images on GitHub.

nbconvert runs the notebook under the ipympl ``widget`` backend, so each figure
is stored as a Jupyter *widget view*. That view is interactive in JupyterLab but
renders blank on GitHub/nbviewer (they don't run the widget JavaScript), which
makes the cell fall back to showing the bare ``<... object at 0x...>`` repr.

ipympl also embeds a plain PNG snapshot alongside the widget view. This script
rewrites each figure output down to just that ``image/png`` and drops the object
reprs, so the committed notebook shows static figures on GitHub while remaining
fully interactive when re-run locally (the source still says ``%matplotlib
widget``). Run it after ``jupyter nbconvert --execute --inplace``.
"""
import nbformat

NB = "example_adi_reduction.ipynb"
nb = nbformat.read(NB, as_version=4)

for cell in nb.cells:
    if cell.get("cell_type") != "code":
        continue
    cleaned = []
    for out in cell.get("outputs", []):
        typ = out.get("output_type")
        if typ == "stream":
            cleaned.append(out)                       # keep print() output as-is
        elif typ in ("display_data", "execute_result"):
            png = out.get("data", {}).get("image/png")
            if png is not None:                       # a figure: keep only the PNG
                cleaned.append(nbformat.v4.new_output(
                    "display_data", data={"image/png": png}))
            # otherwise it is a bare object repr (text/plain only) -> drop it
    cell["outputs"] = cleaned

nb.get("metadata", {}).pop("widgets", None)           # drop now-orphaned widget state
nbformat.write(nb, NB)
print(f"rewrote {NB}: figures reduced to static image/png, object reprs dropped")
