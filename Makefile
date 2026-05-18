.PHONY: setup build run clean

# Default target
.DEFAULT_GOAL := build

# Create .env with current user's UID/GID and HF_TOKEN
setup:
	@echo "UID=$$(id -u)" > .env
	@echo "GID=$$(id -g)" >> .env
	@if [ -z "$$HF_TOKEN" ]; then \
		read -p "Enter your HuggingFace token (HF_TOKEN): " token && \
		echo "HF_TOKEN=$$token" >> .env; \
	else \
		echo "HF_TOKEN=$$HF_TOKEN" >> .env; \
	fi
	@echo "Created .env with UID=$$(id -u) GID=$$(id -g) and HF_TOKEN"

# Build the Docker image (preserves existing HF_TOKEN from .env)
build:
	@if [ -f .env ] && grep -q "^HF_TOKEN=" .env; then \
		export $$(grep "^HF_TOKEN=" .env | xargs) && \
		echo "UID=$$(id -u)" > .env && \
		echo "GID=$$(id -g)" >> .env && \
		echo "HF_TOKEN=$$HF_TOKEN" >> .env; \
	else \
		$(MAKE) setup; \
	fi
	docker compose build

# Run the training environment
run:
	docker compose run --remove-orphans training_env

# Stop and remove all Kaldi containers and images
clean:
	@echo "Stopping and removing Kaldi containers..."
	-docker compose down --remove-orphans
	-docker ps -a --filter "name=kaldi" -q | xargs -r docker stop
	-docker ps -a --filter "name=kaldi" -q | xargs -r docker rm
	@echo "Removing Kaldi images..."
	-docker images --filter "reference=*kaldi*" -q | xargs -r docker rmi -f
	@echo "Cleanup complete."
