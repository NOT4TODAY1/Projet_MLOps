.PHONY: install prepare train clean help run runall

install:
	python -m pip install -r requirements.txt

prepare:
	python main.py --prepare

train:
	python main.py --train

runall:
	python main.py --runall

clean:
	if exist models rmdir /s /q models
	if exist results rmdir /s /q results
	for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

help:
	@echo "install - pip install -r requirements.txt"
	@echo "prepare - python main.py --prepare"
	@echo "train - python main.py --train"

run:
	python main.py
