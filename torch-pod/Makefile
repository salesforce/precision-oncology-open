image_name = torch-pod-aesteva
gcr_image_name = gcr.io/salesforce-research-internal/torch-pod-aesteva

# -------------------------------- Local ---------------------------
build:
	docker build -t $(image_name) .

run:
	docker run -it \
   --publish 8888:8888 \
   --publish 443:443 \
   --name $(image_name) \
   --rm \
   $(image_name):latest

run-bind:
	docker run -it \
   --publish 8080:8080 \
   --publish 443:443 \
   --name $(image_name) \
   --mount type=bind,source="$(shell pwd)",target=/workspace/full \
   --rm \
   $(image_name):latest

attach:
	docker exec -it $(image_name) zsh

local: build run

# -------------------- Docker Image to GCP -------------------------
build-gcr:
	docker build -t $(gcr_image_name) .

run-gcr:
	docker run -it \
   --publish 8080:8080 \
   --publish 443:443 \
   --name $(gcr_image_name) \
   --mount type=bind,source="$(shell pwd)",target=/workspace/full \
   --rm \
   $(gcr_image_name):latest

push-gcr:
	docker push $(gcr_image_name):latest

gcr: build-gcr push-gcr


# -------------------- Torch Pod on GCP -------------------------

deploy-megamem:
	kubectl apply -f deploy-megamem.yaml

deploy-ultramem:
	kubectl apply -f deploy-ultramem.yaml

deploy-no-gpu:
	kubectl apply -f deploy-no-gpu.yaml

deploy-1gpu-16cpu:
	kubectl apply -f deploy-1gpu-16cpu.yaml

deploy-8gpu-60cpu:
	kubectl apply -f deploy-8gpu-60cpu.yaml

deploy-8gpu-70cpu:
	kubectl apply -f deploy-8gpu-70cpu.yaml

service:
	kubectl apply -f service.yaml

pod: service deploy
