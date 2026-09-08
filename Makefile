# CV-localization experiment suite.
#
#   make build          build the Docker image (runs the tests)
#   make smoke          full suite at smoke scale, a few minutes
#   make quick          intermediate scale
#   make paper          the configuration reported in the article
#   make exp07          one experiment, current SCALE
#   make tables         regenerate the LaTeX tables and FINDINGS.md
#   make shards N=4     run N shards in parallel, then merge
#   make merge          merge sharded results and rebuild tables
#   make logs           follow the log of the current scale
#   make findings       print the digest of the current scale
#   make notebook       Jupyter on port 8888
#   make test           unit tests
#   make shell          interactive shell in the container
#   make overnight      tests, smoke pre-check, then the full paper run
#   make clean          wipe results/
#
# Everything honours SCALE, e.g.  make exp07 SCALE=quick
# Local runs without Docker:      make local-exp07 SCALE=smoke

SCALE ?= smoke
IMAGE ?= cvloc-fpa-enkf:latest
N ?= 4
METHODS ?=
EXPERIMENTS_SEL ?=
DOCKER_RUN = docker run --rm -e SCALE=$(SCALE) -e PYTHONHASHSEED=0 \
	-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
	-v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper $(IMAGE)

EXPERIMENTS = exp00_tuning exp01_landscape exp02_baselines exp03_cvproxy \
              exp04_meta exp05_budget exp06_ensemble exp07_kradii \
              exp08_hetero exp09_cycles exp10_cholesky exp11_smoother

.PHONY: build test smoke quick paper all tables findings notebook shell clean \
        help logs shards merge shard-status shard-clean overnight $(EXPERIMENTS)

help:
	@sed -n '2,20p' Makefile

build:
	docker build -t $(IMAGE) .

test:
	$(DOCKER_RUN) python -m pytest tests -q

smoke:
	$(MAKE) all SCALE=smoke

quick:
	$(MAKE) all SCALE=quick

paper:
	$(MAKE) all SCALE=paper

all:
	$(DOCKER_RUN) bash /work/scripts/run_all.sh $(SCALE)

$(EXPERIMENTS):
	$(DOCKER_RUN) python /work/experiments/$@.py $(SCALE)

# Short aliases: make exp07 instead of make exp07_kradii
exp00: exp00_tuning
exp01: exp01_landscape
exp02: exp02_baselines
exp03: exp03_cvproxy
exp04: exp04_meta
exp05: exp05_budget
exp06: exp06_ensemble
exp07: exp07_kradii
exp08: exp08_hetero
exp09: exp09_cycles
exp10: exp10_cholesky
exp11: exp11_smoother

# EXP-00 writes results/frozen_params.json, which the others read. When
# sharding, run it once on its own first so every shard sees the same
# calibrated hyperparameters.
shards: build
	@echo "calibrating hyperparameters once before sharding"
	$(DOCKER_RUN) python /work/experiments/exp00_tuning.py $(SCALE)
	@echo "launching $(N) shards for scale=$(SCALE)"
	@for i in $$(seq 0 $$(( $(N) - 1 ))); do \
		docker run -d --name cvl-shard$$i \
			-e SCALE=$(SCALE) -e SHARD_INDEX=$$i -e SHARD_COUNT=$(N) \
			-e METHODS="$(METHODS)" \
			-e EXPERIMENTS="$(if $(EXPERIMENTS_SEL),$(EXPERIMENTS_SEL),exp01_landscape exp02_baselines exp03_cvproxy exp04_meta exp05_budget exp06_ensemble exp07_kradii exp08_hetero exp09_cycles exp10_cholesky exp11_smoother)" \
			-e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 \
			-v $(PWD)/results:/work/results $(IMAGE) \
			bash /work/scripts/run_all.sh ; \
	done
	@echo "follow with: docker logs -f cvl-shard0   (or any index)"
	@echo "when all are done: make merge SCALE=$(SCALE) N=$(N)"

merge:
	docker run --rm -e SCALE=$(SCALE) -e SHARD_COUNT=$(N) \
		-v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper $(IMAGE) \
		bash -lc "python /work/scripts/merge_shards.py $(SCALE) && \
		          python /work/scripts/make_tables.py $(SCALE)"

shard-status:
	@docker ps -a --filter "name=cvl-shard" --format "table {{.Names}}\t{{.Status}}"

shard-clean:
	-@docker rm -f $$(docker ps -aq --filter "name=cvl-shard") 2>/dev/null || true

tables:
	$(DOCKER_RUN) python /work/scripts/make_tables.py $(SCALE)

findings:
	@cat results/FINDINGS_$(SCALE).md 2>/dev/null || \
		echo "no digest yet; run 'make tables SCALE=$(SCALE)'"

logs:
	@tail -f results/run_$(SCALE).log

notebook:
	docker compose up notebook

shell:
	docker run --rm -it -e SCALE=$(SCALE) \
		-v $(PWD)/results:/work/results -v $(PWD):/work/src $(IMAGE) bash

overnight:
	nohup bash scripts/overnight.sh $(SCALE) > results/nohup_$(SCALE).out 2>&1 &
	@echo "started in the background; follow with: tail -f results/nohup_$(SCALE).out"

clean:
	rm -rf results/EXP-* results/run_*.log results/FINDINGS_*.md \
	       results/frozen_params.json

# Local execution without Docker, for quick iteration.
local-%:
	PYTHONPATH=$(PWD):$(PWD)/experiments \
		PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MPLBACKEND=Agg SCALE=$(SCALE) \
		python3 experiments/$*.py $(SCALE)

local-all:
	PYTHONPATH=$(PWD):$(PWD)/experiments \
		PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
		bash scripts/run_all.sh $(SCALE)

local-test:
	PYTHONPATH=$(PWD) python3 -m pytest tests -q
