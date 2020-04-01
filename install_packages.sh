/#!/usr/bin/env bash
pip3 install -r requirements.txt
python3 -m nltk.downloader all
python3 -m spacy download en_core_web_sm
