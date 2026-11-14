# HydroprocessingMCP

A [FastMCP](https://gofastmcp.com) server (and plain Python library) for refinery
hydroprocessing engineering: hydrotreaters (naphtha, kerosene, ULSD, VGO/FCC pretreat) and
hydrocrackers. Its layout follows [ChemMCP](https://github.com/OSU-NLP-Group/ChemMCP): tools
are grouped by domain, each one is usable from an LLM client or from Python, and
[RDKit](https://github.com/rdkit/rdkit) backs the molecule tools.

| Section | Namespace | What it does |
|---|---|---|
| 1. Historical analysis | `analysis_*` | Load time-series data from CSV/Excel/Parquet, SQL or PI Web API. Map tags, clean data, detect steady state, and apply hydroprocessing equations over a whole dataset. You can add your own equations or override the built-in ones. |
| 2. Reactor loading | `loading_*` | Vessel and head geometry; layer-by-layer loading plans (depth, volume, mass, sock vs dense, outage, bottom-head fill); order quantities and package counts; Ergun ΔP screening. |
| 3. Forecasting | `forecast_*` | Fit deactivation of normalized WABT (linear, quadratic, piecewise or exponential) and predict end of run with an uncertainty band. Forecast variables and yields (Holt, ARIMA, linear). Run what-if scenarios. |
| 4. Proposals | `proposals_*` | Catalyst library with performance values. Define catalyst systems (type, quantity and location per bed and layer) and calibrate a kinetic model on the existing unit. Predict SOR WABT, product S/N/aromatics, H₂ consumption, yields and cycle length, then compare and rank proposals. |
| Chemistry | `chemistry_*` | RDKit tools for model compounds (DBT, 4,6-DMDBT, quinoline, carbazole and others): properties, S/N wt%, H₂ stoichiometry for HDS/HDN/HDA pathways, and reaction balance checks. |

The server has 53 tools and three resources (`hydromcp://equations`, `hydromcp://catalysts`,
`hydromcp://units`). DataFrames and fitted models stay on the server, and tools pass
`dataset_id`, `model_id` and `calibration_id` references, so results sent to the LLM stay small.

## Install

```bash
uv sync                       # Python >= 3.11; installs fastmcp, rdkit, pandas, statsmodels, ...
uv run pytest                 # 38 tests
uv run python scripts/generate_sample_data.py   # (re)build data/sample/synthetic_ulsd.csv
```

> Intel macOS: `pyproject.toml` sets `tool.uv.required-environments`. This makes uv choose
> an RDKit release that still ships x86_64 macOS wheels.

### Connect to Claude

Claude Code:
```bash
claude mcp add hydromcp -- uv run --directory /path/to/HydroprocessingMCP hydromcp
```

Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "hydromcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/HydroprocessingMCP", "hydromcp"],
      "env": { "PI_WEB_API_URL": "https://your-pi-server/piwebapi", "PI_DATA_SERVER": "PISRV01" }
    }
  }
}
```

To run it as an HTTP server, use `uv run hydromcp --http` (serves on 127.0.0.1:8000). To open
the inspector, use `uv run fastmcp dev src/hydromcp/server.py`.

### Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `HYDROMCP_DATA_DIR` | Folder with `catalysts/ units/ proposals/ equations/ tagmaps/` (default `./data`) |
| `HYDROMCP_OUTPUT_DIR` | Export folder (default `<data>/output`) |
| `HYDROMCP_SQL_<ALIAS>` | SQLAlchemy URL for a named connection, e.g. `HYDROMCP_SQL_LIMS=mssql+pyodbc://...`. Tools refer to it as `connection="lims"`, so credentials never pass through the chat. |
| `PI_WEB_API_URL`, `PI_DATA_SERVER` | PI Web API base URL and default Data Archive |
| `PI_WEB_API_USER` / `PI_WEB_API_PASSWORD` or `PI_WEB_API_TOKEN` | Basic or bearer auth |
| `PI_WEB_API_VERIFY_SSL` | Set to `false` to skip TLS verification (for internal CAs) |

SQL access is read-only: only single `SELECT`/`WITH` statements are accepted. Install drivers
with `uv sync --extra sql` (pyodbc, psycopg).

## Units

Field units are used throughout: °F, psia (gauge inputs end in `_psig`), BPD, SCF/bbl at
60 °F and 1 atm, wppm, lb, ft, lb/ft³. `hydromcp.core.units` has pint-based `convert()` plus
helpers (`psig_to_psia`, `f_to_k`, `api_to_sg`, …).

## Typical workflow (what to ask Claude)

1. "Load `data/sample/synthetic_ulsd.csv` with the `example` tag map."
2. "Compute bed WABTs, then the catalyst-weighted reactor WABT, LHSV, H₂ partial pressure,
   H₂ consumption and normalized WABT (reference: 12,000 wppm feed S, 8 wppm product,
   900 psia, LHSV 1.8). Then add catalyst age."
3. "Fit the deactivation trend and tell me when we hit the 760 °F EOR."
4. "Calibrate ULSD-1 on the last 60 days, then compare `ULSD-1:current`, `PROP-COMO-HA`
   and `PROP-STACKED` at 8 wppm product sulfur."
5. "What if feed rate goes to 40,000 BPD?"

The same functions work as a library:

```python
from hydromcp.core.library import get_library
from hydromcp.proposals.predict import calibrate, predict
lib = get_library(); unit = lib.unit("ULSD-1")
cal = calibrate(unit, lib.proposal("ULSD-1:current"), unit.design_basis)
predict(lib.proposal("PROP-COMO-HA"), cal, unit.design_basis, product_sulfur_target_wppm=8)
```

## Data files

All data files are YAML (JSON also works) and are validated by the Pydantic models in
`src/hydromcp/core/schemas.py`.

- **`data/catalysts/*.yaml`**: `Catalyst` entries. Each has a function (inert, grading, hdm,
  hds, hdn, hck, …), shape and size, sock and dense loaded densities, void fractions,
  **RVA per reaction** (hds/hdn/hda/hck), activation energy, reaction order, H₂PP exponent,
  deactivation rate (°F/month), H₂ consumption factor, max temperature, hydrocracking yield
  selectivity, package size and cost. The examples are generic and illustrative; replace
  them with your vendor data.
- **`data/units/*.yaml`**: a `ProcessUnit` with its reactors (ID, head type) and beds
  (straight-side height, bottom-head fill, internals area, minimum outage), plus
  `current_loading`, `design_basis`, max WABT and the product sulfur spec.
- **`data/proposals/*.yaml`**: a `Proposal`, i.e. a catalyst system. Beds are listed in flow
  order and layers top to bottom. Each layer has a `catalyst_id`, a `loading_method`
  (sock or dense) and exactly one of `depth_ft`, `volume_ft3`, `mass_lb` or `bed_fraction`.
  Layers are stacked from the bed support upward. Material below the bottom tangent line
  fills the head using the head's real shape.
- **`data/tagmaps/*.yaml`**: maps raw tags to canonical names, with optional unit, scale and offset.
- **`data/equations/`**: your equations. See [data/equations/README.md](data/equations/README.md).

## Built-in equations

Each can be replaced by a user equation with the same name:

- **Temperature:** `wabt_bed`, `wabt_bed_multipoint`, `wabt_reactor`, `bed_temperature_rise`
- **Space velocity:** `lhsv`, `whsv`
- **Hydrogen:** `treat_gas_ratio`, `h2_partial_pressure`, `h2_consumption`
- **Kinetics:** `removal_pct`, `apparent_rate_constant`, `normalized_wabt`,
  `required_temperature_change`
- **Catalyst:** `catalyst_life`
- **Balance:** `mass_balance_closure`, `liquid_yield`, `sulfur_removed_lb_day`
- **Properties:** `api_gravity`, `specific_gravity`, `cetane_index_d4737`
- **Hydraulics:** `pressure_drop_normalized`

## Model basis and limitations (Section 4)

- **Kinetics:** the model runs layer-by-layer plug flow. HDS uses nth-order kinetics
  (default n = 1.5); HDN, HDA and HCK use first order. Rate constants follow Arrhenius, with
  an H₂PP power-law term:
  `k = k_ref · RVA · exp[−Ea/R (1/T − 1/T_ref)] · (PH₂/PH₂,ref)^α`. Every layer is solved at
  its own LHSV. Per-bed temperature offsets from WABT are optional.
- **Calibration:** `k_ref` is fitted to the unit's own measured performance with its existing
  loading. Catalyst age is corrected back to a fresh-catalyst basis. RVAs therefore need to
  be relative to the same reference catalyst.
- **Deactivation:** each catalyst's °F/month rate is treated as an effective temperature
  loss. The cycle is simulated month by month until the required WABT reaches the max WABT,
  so mixed systems age realistically.
- **H₂ and yields:** H₂ consumption is stoichiometric (S, N, aromatics saturated,
  hydrocracking conversion) times the catalyst H₂ factors. Default mol-H₂ ratios can be
  overridden (for example from `chemistry_h2_for_feed`). Yields are wt% of feed: H₂S, NH₃,
  hydrocracked cuts from the catalyst selectivities, and treated liquid. The mass balance
  closes to 100 + H₂ wt%.
- **Simplifications:** the model has no explicit H₂S or nitrogen inhibition, no
  aromatics-equilibrium limit, no heat balance (WABT is an input or is solved for), and no
  trickle-flow wetting or hydrodynamics. The Ergun ΔP is a clean-bed screening index.
  Any of these can be added as user equations or by extending `proposals/kinetics.py`.

## Layout

```
src/hydromcp/
  server.py            root FastMCP server; mounts the five namespaces
  core/                equation registry, units, schemas, library (YAML), session store
  connectors/          files, sql, pi (PI Web API), tagmap
  analysis/            built-in equations, time-series operations
  loading/             geometry, loading plans, hydraulics
  forecast/            deactivation/EOR, time-series forecasts, what-if
  proposals/           kinetic model, calibration, prediction, comparison
  chemistry/           RDKit molecule utilities, model compounds, stoichiometry
  tools/               MCP tool wrappers, one sub-server per section
scripts/generate_sample_data.py   synthetic ULSD historian export, generated from the model
tests/                 unit tests plus an end-to-end MCP workflow test
```
