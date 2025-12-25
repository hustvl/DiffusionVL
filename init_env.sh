pip install -r requirements.txt -y
cd eval/lmms-eval
pip install --no-deps -U -e .
cd ../..
cd train
pip install --no-deps -U -e .