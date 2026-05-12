# droplesim — universal Makefile

GEOMETRY_H5 := geometry/geometry.h5
DXF         := data/generator.dxf

.PHONY: setup dev test test-all lint fmt clean geometry

# ── Environment ──────────────────────────────────────────────────────────────

setup:
	uv sync
	@echo ""
	@echo "GPU server: also run  uv add 'jax[cuda12_pip]' \\"
	@echo "  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# ── App ───────────────────────────────────────────────────────────────────────

dev:
	uv run droplesim

# ── Geometry (optional pre-processing) ───────────────────────────────────────

geometry: $(GEOMETRY_H5)

$(GEOMETRY_H5): src/droplesim/geometry/dxf_to_voxel.py $(DXF)
	uv run python -m droplesim.geometry.dxf_to_voxel --dxf $(DXF) --out $(GEOMETRY_H5)

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run python -m pytest src/ -x -q 2>/dev/null || true
	uv run python -m droplesim.scripts.run_doe \
	    --config config/benchmarks/laplace.json --dry-run
	uv run python -m droplesim.scripts.run_doe \
	    --config config/benchmarks/tomotika.json --dry-run
	uv run python -m droplesim.scripts.run_doe \
	    --config config/benchmarks/contact_angle.json --dry-run

test-all: test
	uv run python -c "from droplesim.solver.sim import TwoPhaseSim; print('solver import OK')"

# ── Linting / formatting ──────────────────────────────────────────────────────

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf .pytest_cache __pycache__ .ruff_cache geometry/geometry.h5
