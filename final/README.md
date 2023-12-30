# DLCV Final Project ( Visual Queries 2D Localization Task )
Team 4: Baseliner

# To train
Please modify the three directory path of "base_dataset.py" to your dataset path.
```shell
python3 train_anchor.py --cfg ./config/train.yaml
```

<!-- # To extract query image from clip and json
```shell
python3 extract_img_clip.py --annot-path <json file> --clips-root <clip folder> --vis-save-root <output image folder>
``` -->

# To inference
Download the trained model and dataset using this command:
```shell
bash download.sh
```

And also put `clips/*`, `vq_test_unannotated.json` under `student_data/`.

You can use shell scripts to run inference on validation set and test set.
```shell
bash val.sh # for validation set and do evaluation
bash test.sh # for test set
```

The results of test set should be in `output/ego4d_vq2d/eval/inference_result`.


---
Alternative Method: (validation set)

change the model cpt_path in config/val.yaml first if needed 
```shell
python3 inference_predict.py --cfg ./config/val.yaml
python3 inference_results.py --cfg ./config/val.yaml
```
the result is in output/ego4d_vq2d/val/validate/inference_cache_val_results.json, then run
```shell
cd evaluation
python3 evaluate_vq.py --gt-file <vq_val.json> --pred-file <pred json file>
```


<!-- ---

# How to run your code?
* TODO: Please provide the scripts for TAs to reproduce your results, including training and inference. For example, 

```shell script=
bash train.sh <Path to clips folder> <Path to annot file>
bash inference.sh <Path to clips folder> <Path to annot file> <Path to output json file>
```

# Usage
To start working on this final project, you should clone this repository into your local machine by the following command:

    git clone https://github.com/ntudlcv/DLCV-Fall-2023-Final-2-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1TsR0l84wWNNWH7HaV-FEPFudr3o9mVz29LZQhFO22Vk/edit?usp=sharing) to view the slides of Final Project - Visual Queries 2D Localization Task. **The introduction video for final project can be accessed in the slides.**

# Visualization
We provide the code for visualizing your predicted bounding box on each frame. You can run the code by the following command:

    python3 visualize_annotations.py --annot-path <annot-path> --clips-root <clips-root> --vis-save-root <vis-save-root>

Note that you should replace `<annot-path>`, `<clips-root>`, and `<vis-save-root>` with your annotation file (e.g. `vq_val.json`), the folder contains all clips, and the output folder of the visualization results, respectively.

# Evaluation
We also provide the evaluation for you to check the performance (stAP) on validation set. You can run the code by the following command:

    cd evaluation/
    python3 evaluate_vq.py --gt-file <gt-file> --pred-file <pred-file>

Note that you should replace `<gt-file>` with your val annotation file (e.g. `vq_val.json`) and replace `<pred-file>` with your output prediction file (e.g. `pred.json`)  

# Submission Rules
### Deadline
112/12/28 (Thur.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under `[Final challenge 2] VQ2D discussion` section in NTU Cool Discussion -->
