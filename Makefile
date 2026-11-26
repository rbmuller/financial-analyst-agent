COMPOSE := docker compose
RUN := $(COMPOSE) run --rm app

.PHONY: up down reset migrate ingest index search ask slack eval eval-retrieval psql test lint

# No prerequisite on purpose: an edited .env must never be overwritten.
.env:
	cp .env.example .env

up: .env ## Start Postgres and apply migrations
	$(COMPOSE) up -d --wait db
	$(RUN) python -m db.migrate

down: ## Stop containers (data is kept)
	$(COMPOSE) down

reset: ## Stop containers and DELETE the database volume
	$(COMPOSE) down -v

migrate: ## Apply pending migrations
	$(RUN) python -m db.migrate

ingest: ## Ingest SEC data (extra flags: make ingest ARGS="--tickers NKE --only xbrl")
	$(RUN) python -m ingestion.cli $(ARGS)

index: ## Chunk and embed ingested filings (ARGS="--rebuild" to start over)
	$(RUN) python -m rag.cli index $(ARGS)

search: ## Search filings: make search Q="gross margin decline" ARGS="--ticker NKE --rerank"
	$(RUN) python -m rag.cli search "$(Q)" $(ARGS)

ask: ## Ask the agent: make ask Q="Why did Nike's gross margin fall?" ARGS="--rerank"
	$(RUN) python -m agent.cli "$(Q)" $(ARGS)

slack: ## Run the Slack bot (Socket Mode); stop with Ctrl+C
	$(RUN) python -m slack_app.main

eval-retrieval: ## Retrieval eval (free, no LLM calls)
	$(RUN) python -m evals.run retrieval

eval: ## Full eval: retrieval + agent answers graded by a judge (calls the Anthropic API)
	$(RUN) python -m evals.run retrieval
	$(RUN) python -m evals.run judge-check
	$(RUN) python -m evals.run answers $(ARGS)

psql: ## Open psql as the admin user
	$(COMPOSE) exec db sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

test: ## Run unit + integration tests
	$(RUN) pytest

lint: ## Ruff + mypy
	$(RUN) sh -c "ruff check . && ruff format --check . && mypy ."
