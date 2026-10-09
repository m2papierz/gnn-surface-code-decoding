.PHONY: install setup fmt lint check test clean

UV  ?= uv
RUN := $(UV) run --locked

# ── Install ──────────────────────────────────────────────────────────────────

install:
	$(UV) sync --locked

# ── Setup (first-time contributor) ───────────────────────────────────────────

setup: install
	$(RUN) pre-commit install
	@echo "  ✓ Pre-commit hooks installed"

# ── Format ───────────────────────────────────────────────────────────────────

fmt:
	$(RUN) ruff check --select I --fix src/ scripts/ tests/
	$(RUN) ruff format src/ scripts/ tests/

# ── Lint ─────────────────────────────────────────────────────────────────────

lint:
	$(RUN) ruff check src/ scripts/ tests/

# ── Type-check ───────────────────────────────────────────────────────────────

check:
	$(RUN) pyright

# ── Test ─────────────────────────────────────────────────────────────────────

test:
	$(RUN) pytest

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf dist/ build/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
