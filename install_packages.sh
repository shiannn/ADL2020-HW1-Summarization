#!/usr/bin/env bash
pip install -r requirements.txt
python -m nltk.downloader all
python -m spacy download en_core_web_sm
