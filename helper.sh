python3 -m venv venv

source venv/bin/activate

python setup_nltk.py

python -m spacy download en_core_web_sm

pip freeze > requirements.txt