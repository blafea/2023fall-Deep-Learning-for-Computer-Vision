#!bin/bash
python3 inference_predict.py --eval --cfg ./config/eval.yaml
python3 inference_results.py --eval --cfg ./config/eval.yaml