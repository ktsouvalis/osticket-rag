.PHONY: api-up api-down create-collection load-initial update init-state build-updater build-load build-create

DOCKER_COMPOSE ?= docker compose

api-up:
	$(DOCKER_COMPOSE) up -d --build rag-api

api-down:
	$(DOCKER_COMPOSE) down

create-collection:
	$(DOCKER_COMPOSE) --profile manual run --rm rag-create-collection

load-initial:
	$(DOCKER_COMPOSE) --profile manual run --rm rag-load-initial

update:
	$(DOCKER_COMPOSE) --profile manual run --rm rag-updater

init-state:
	@test -f .milvus_update_state.json || echo '{"last_activity_ts": 0, "last_faq_id": 0}' > .milvus_update_state.json

build-updater:
	$(DOCKER_COMPOSE) build rag-updater

build-load:
	$(DOCKER_COMPOSE) build rag-load-initial

build-create:
	$(DOCKER_COMPOSE) build rag-create-collection
