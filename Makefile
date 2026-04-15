# Primary registry — Harbor (used by docker-server and all homelab hosts)
REGISTRY    := harbor.homeserverlocal.com/mcp
IMAGE       := $(REGISTRY)/infragraph
TAG         ?= latest

# Legacy local registry (dev-workstation only; not resolvable from docker-server)
LOCAL_REGISTRY := docker-local.homeserverlocal.com
LOCAL_IMAGE    := $(LOCAL_REGISTRY)/infragraph/infragraph

DEPLOY_HOST := application@192.168.0.50
DEPLOY_DIR  := /opt/infragraph

# ── build & push from Mac (linux/amd64 via buildx, direct push — no local load) ──

.PHONY: build
build:
	docker buildx build \
		--platform linux/amd64 \
		--tag $(IMAGE):$(TAG) \
		--tag $(LOCAL_IMAGE):$(TAG) \
		--push \
		.

# ── deploy: copy compose + .env to devops-server, pull from Harbor, restart ───

.PHONY: deploy
deploy:
	ssh $(DEPLOY_HOST) "sudo mkdir -p $(DEPLOY_DIR) && sudo chown application:application $(DEPLOY_DIR)"
	scp docker-compose.yml $(DEPLOY_HOST):$(DEPLOY_DIR)/docker-compose.yml
	scp .env               $(DEPLOY_HOST):$(DEPLOY_DIR)/.env
	ssh $(DEPLOY_HOST) "cd $(DEPLOY_DIR) && \
		docker compose pull && \
		docker compose up -d --remove-orphans"

# ── full workflow ─────────────────────────────────────────────────────────────

.PHONY: release
release: build deploy
	@echo "Released $(IMAGE):$(TAG) and $(LOCAL_IMAGE):$(TAG) to $(DEPLOY_HOST)"

# ── registry login (run once) ─────────────────────────────────────────────────

.PHONY: login
login:
	docker login $(REGISTRY) -u admin

# ── logs ──────────────────────────────────────────────────────────────────────

.PHONY: logs
logs:
	ssh $(DEPLOY_HOST) "cd $(DEPLOY_DIR) && docker compose logs -f infragraph-mcp"

# ── local dev (stdio MCP for Claude Code on this Mac) ────────────────────────

.PHONY: run-local
run-local:
	MODE=stdio python -m infragraph.mcp.server
