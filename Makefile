.PHONY: install help api-up api-down create-collection load-initial update build-updater build-load build-create install

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  api-up            Build and start the API container (rag-api)"
	@echo "  api-down          Stop and remove containers"
	@echo "  create-collection Drops and recreates the Milvus collection (manual profile)"
	@echo "  load-initial      Creates and stores the embeddings (manual profile)"
	@echo "  check-updates     Check for updates without applying them (manual profile)"
	@echo "  update            Incremental update of Milvus collection based on the state file (manual profile)"
	@echo "  build-updater     In case of dependencies changes, build the rag-updater image"
	@echo "  build-load        In case of dependencies changes, build the rag-load-initial image"
	@echo "  build-create      In case of dependencies changes, build the rag-create-collection image"
	@echo "  install           WARNING: This initializes the database. It runs: create-collection + load-initial + api-up"

DOCKER_COMPOSE ?= docker compose

api-up:
	$(DOCKER_COMPOSE) up -d --build rag-api

api-down:
	$(DOCKER_COMPOSE) down

create-collection:
	$(DOCKER_COMPOSE) --profile manual run --rm --build rag-create-collection

load-initial:
	$(DOCKER_COMPOSE) --profile manual run --rm --build rag-load-initial

check-updates:
	$(DOCKER_COMPOSE) --profile manual run --rm --build rag-checker

update:
	$(DOCKER_COMPOSE) --profile manual run --rm --build rag-updater

build-updater:
	$(DOCKER_COMPOSE) build rag-updater

build-load:
	$(DOCKER_COMPOSE) build rag-load-initial

build-create:
	$(DOCKER_COMPOSE) build rag-create-collection

install: create-collection load-initial api-up
	@echo "Installation complete. RAG API is running."
