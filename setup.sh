#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt