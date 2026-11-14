"""HydroprocessingMCP server: mounts the section servers and exposes browsable resources.

Run:  uv run hydromcp            (stdio, for Claude Desktop / Claude Code)
      uv run hydromcp --http     (streamable HTTP on 127.0.0.1:8000)
"""

from __future__ import annotations

import sys

from fastmcp import FastMCP

from hydromcp import __version__, config
from hydromcp.analysis import calcs  # noqa: F401  (registers built-in equations)
from hydromcp.core.equations import REGISTRY, load_user_equations
from hydromcp.core.library import get_library
from hydromcp.tools import analysis, chemistry, forecast, loading, proposals

INSTRUCTIONS = """Hydroprocessing (hydrotreating / hydrocracking) engineering tools. Field units:
degF, psia/psig, BPD, SCF/bbl, wppm, lb, ft.
Workflow:
1. analysis_*: load historian/LIMS data (file, SQL, PI), clean, detect steady state, apply
   equations (WABT, LHSV, H2 consumption, normalized WABT ...), add catalyst age.
2. loading_*: reactor geometry and layer-by-layer catalyst loading plans for a unit.
3. forecast_*: fit deactivation of normalized WABT, predict end-of-run, forecast
   variables, run what-if scenarios on a calibrated unit.
4. proposals_*: catalyst library, define catalyst systems (proposals), calibrate the
   kinetic model on the existing unit (proposals_calibrate_unit), then predict and
   compare proposals.
5. chemistry_*: RDKit model-compound properties and H2 stoichiometry.
Large data stays on the server, referenced by dataset_id / model_id / calibration_id."""

mcp = FastMCP("HydroprocessingMCP", instructions=INSTRUCTIONS, version=__version__)
mcp.mount(analysis.mcp, namespace="analysis")
mcp.mount(loading.mcp, namespace="loading")
mcp.mount(forecast.mcp, namespace="forecast")
mcp.mount(proposals.mcp, namespace="proposals")
mcp.mount(chemistry.mcp, namespace="chemistry")

load_user_equations(config.data_dir() / "equations")


@mcp.resource("hydromcp://equations", mime_type="application/json")
def equations_resource() -> list[dict]:
    """Catalog of all registered equations."""
    return [e.to_dict() for e in REGISTRY.list()]


@mcp.resource("hydromcp://catalysts", mime_type="application/json")
def catalysts_resource() -> list[dict]:
    """The catalyst library."""
    return [c.model_dump(exclude_none=True) for c in get_library().catalysts()]


@mcp.resource("hydromcp://units", mime_type="application/json")
def units_resource() -> list[dict]:
    """Process unit definitions."""
    return [u.model_dump(exclude_none=True) for u in get_library().units()]


def main() -> None:
    if "--http" in sys.argv:
        mcp.run(transport="http", host="127.0.0.1", port=8000)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
