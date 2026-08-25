# Convenience targets for the Docker + Kubernetes workflow.
# Run `make help` to list them.

IMAGE_TRAIN ?= mlops-train:v1
IMAGE_SERVE ?= mlops-serve:v1
NAMESPACE   ?= ml-training

.PHONY: help lint test build-train build-serve build run-train run-serve k8s-up k8s-down

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

lint:  ## Run ruff lint + format check
	ruff check src tests
	ruff format --check src tests

test:  ## Run unit tests
	pytest tests -v

build-train:  ## Build the training image
	docker build -f docker/Dockerfile.train -t $(IMAGE_TRAIN) .

build-serve:  ## Build the serving image
	docker build -f docker/Dockerfile.serve -t $(IMAGE_SERVE) .

build: build-train build-serve  ## Build both images

run-train:  ## Run training locally with mounted volumes
	docker run --rm \
	  -v $(PWD)/data:/app/data \
	  -v $(PWD)/checkpoints:/app/checkpoints \
	  $(IMAGE_TRAIN)

run-serve:  ## Run the serving container locally on :8080
	docker run --rm -p 8080:8080 \
	  -v $(PWD)/checkpoints:/app/checkpoints \
	  $(IMAGE_SERVE)

k8s-up:  ## Apply namespace, config and training job
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/pvc.yaml
	kubectl apply -f k8s/training-job.yaml

k8s-down:  ## Delete everything in the project namespace
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
