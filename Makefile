install:
	python -m pip install -r requirements.txt

run:
	python main.py

clean:
	rm -rf results models __pycache__
