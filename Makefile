PY=python3

.PHONY: setup data train eval decode profile test clean

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data --prepare

train:
	$(PY) -m src.train --config configs/train.yaml

eval:
	$(PY) -m src.eval --split val

decode:
	$(PY) -m src.decoding --strategy beam --beam_size 4 --length_penalty 0.6

profile:
	$(PY) -m src.train --config configs/train.yaml --profile

test:
	pytest -v tests/

clean:
	rm -rf outputs/ checkpoints/ .pytest_cache __pycache__ src/__pycache__ src/models/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
