COMPETITION := playground-series-s6e6
DATA_DIR   := data
TOKEN_FILE := .kaggle/access_token

SUBMISSION_FILE ?= outputs/submissions/submission.csv
SUBMISSION_MSG  ?= "benchmark: stacked LGBM+XGB+CatBoost with LogisticRegression meta"
CONFIG          ?= config/config.yaml
RUN_NAME        ?=

.PHONY: all install download train predict submit test lint format clean

all: install download train submit
	@echo ""
	@echo "========================================================"
	@echo "  Happy path complete! Check leaderboard above."
	@echo "========================================================"

install: .uv_sync
	uv pip install -e .
	@$(MAKE) _ensure_kaggle_auth
	@echo ""
	@echo "All set. Run 'make download' to fetch the competition data."

download:
	@mkdir -p $(DATA_DIR); \
	$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Downloading $(COMPETITION) data..."; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions download \
		-c $(COMPETITION) -p $(DATA_DIR) 2>&1 || { \
		exit_code=$$?; \
		echo ""; \
		echo "================================================================"; \
		echo " Download failed (403 Forbidden)."; \
		echo ""; \
		echo " Possible causes:"; \
		echo "   1. You haven't joined the competition yet."; \
		echo "      Go to the page and click 'Join' / 'Accept Rules':"; \
		echo "      https://www.kaggle.com/competitions/$(COMPETITION)"; \
		echo ""; \
		echo "   2. Your API token may be stale."; \
		echo "      Regenerate at https://www.kaggle.com/settings"; \
		echo "      then update $(TOKEN_FILE)"; \
		echo "================================================================"; \
		exit $$exit_code; \
	}; \
	echo "Extracting..."; \
	cd $(DATA_DIR) && unzip -o $(COMPETITION).zip && rm -f $(COMPETITION).zip; \
	echo "  Data ready in $(DATA_DIR)/"

train:
	@uv run python scripts/train.py --config $(CONFIG) $(if $(RUN_NAME),--run-name $(RUN_NAME),) $(ARGS)

predict:
	@uv run python scripts/predict.py $(ARGS)

submit:
	@$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	[ ! -f $(SUBMISSION_FILE) ] && { echo "ERROR: $(SUBMISSION_FILE) not found — run 'make train' or 'make predict' first"; exit 1; }; \
	echo "Submitting $(SUBMISSION_FILE) to $(COMPETITION)..."; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions submit \
		-c $(COMPETITION) \
		-f $(SUBMISSION_FILE) \
		-m "$(SUBMISSION_MSG)" && \
	echo "" && \
	echo "✓ Submitted! Checking leaderboard..." && \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions leaderboard \
		-c $(COMPETITION) --show

test:
	@uv run pytest tests/ -v $(ARGS)

lint:
	@uv run ruff check src/ scripts/ tests/

format:
	@uv run ruff format src/ scripts/ tests/ --check

format-fix:
	@uv run ruff format src/ scripts/ tests/

.uv_sync: pyproject.toml uv.lock
	uv sync --extra dev
	@touch .uv_sync

_ensure_kaggle_token:
	@mkdir -p .kaggle; \
	PLACEHOLDER="KGAT_your-kaggle-api-token-here"; \
	TOKEN=""; \
	\
	if [ -n "$$KAGGLE_API_TOKEN" ]; then \
		TOKEN="$$KAGGLE_API_TOKEN"; \
		if [ ! -f $(TOKEN_FILE) ] || [ "$$(cat $(TOKEN_FILE))" != "$$TOKEN" ]; then \
			printf '%s' "$$TOKEN" > $(TOKEN_FILE); \
			chmod 600 $(TOKEN_FILE); \
		fi; \
	elif [ -f $(TOKEN_FILE) ] && [ -s $(TOKEN_FILE) ] && ! grep -q "$$PLACEHOLDER" $(TOKEN_FILE) 2>/dev/null; then \
		TOKEN=$$(cat $(TOKEN_FILE)); \
	elif [ -f ~/.kaggle/kaggle.json ] && ! grep -q "your-kaggle-username" ~/.kaggle/kaggle.json 2>/dev/null; then \
		echo "Using legacy credentials from ~/.kaggle/kaggle.json"; \
		exit 0; \
	else \
		if [ -f .kaggle/access_token.example ]; then \
			cp .kaggle/access_token.example $(TOKEN_FILE); \
		else \
			printf 'KGAT_your-kaggle-api-token-here\n' > $(TOKEN_FILE); \
		fi; \
		chmod 600 $(TOKEN_FILE); \
		echo ""; \
		echo " Token template written to $(TOKEN_FILE)."; \
		echo ""; \
		echo " To configure:"; \
		echo "   1. Go to https://www.kaggle.com/settings"; \
		echo "   2. Under API, click 'Create New Token'"; \
		echo "   3. Copy the token (starts with KGAT_)"; \
		echo "   4. Paste it into $(TOKEN_FILE) (and nothing else)"; \
		echo "   5. Run 'make download' again to verify"; \
		exit 1; \
	fi

_ensure_kaggle_auth: _ensure_kaggle_token
	@TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Verifying..." && \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions list >/dev/null 2>&1 && \
	echo "  Authenticated successfully." || \
	{ echo "  WARNING: Authentication check failed."; exit 1; }

clean:
	rm -f .uv_sync
