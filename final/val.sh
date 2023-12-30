#!bin/bash
python3 inference_predict.py --cfg ./config/val.yaml
python3 inference_results.py --cfg ./config/val.yaml
python3 evaluation/evaluate_vq.py \
    --gt-file student_data/vq_val.json \
    --pred-file output/ego4d_vq2d/val/validate/inference_cache_val_results.json \