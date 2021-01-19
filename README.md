# PReFIL for DVQA and FigureQA

This repository provides code and pretrained models for the PReFIL algorithm described in the <a href="https://wacv20.wacv.net/" target="_blank">WACV 2020</a> Paper:

**[Answering Questions about Data Visualizations using Efficient Bimodal Fusion](https://arxiv.org/abs/1908.01801)**
 <br>
 <a href="https://kushalkafle.com/" target="_blank">Kushal Kafle</a>,
  <a href="https://scholar.google.com/citations?user=RzAjx8UAAAAJ&hl=en" target="_blank">Robik Shrestha</a>,
 <a href="https://research.adobe.com/person/brian-price/" target="_blank">Brian Price</a>,
<a href="https://research.adobe.com/person/scott-cohen/" target="_blank">Scott Cohen</a>,
<a href="http://www.chriskanan.com/" target="_blank">Christopher Kanan</a>

<div align="center">
  <img src="https://kushalkafle.com/images/prefil.png" width="450px">
</div>

We made some minor optimizations to the code (No hyperparameters are changed from the paper). 
As a result, the numbers obtained from this repo are slightly better than reported in the paper. 

| Paper Vs. Repo     |               |  Paper | Repo  |    
|----------|---------------|---|---|---|  
|    DVQA  |  |   |   |   |  
|          | Test Familiar | 96.37  | 96.93  |    
|          | Test Novel    | 96.53 |  96.99 |   
| FigureQA |               |   |   |   |  
|          | Validation 1  | 94.84  | 96.56  |  
|          | Validation 2  | 93.26 | 95.31  |   


# Requirements and Repo Setup

The repo has minimal requirements. All you should really need are `CUDA 10.2`, `nltk` and `pytorch 1.5.1` 
However, we also provide the exact conda environment that we tested the code on. To replicate, simply edit the `prefix` parameter 
in the `requirement.yml` and simply run `conda env create -f requirements.yml`, followed by `conda activate prefil`. 

# Setting up Data

Head over to the `README` under data to download and setup data and pretrained models

# Training, Resuming and Evaluating models

The `run_cqa.py` provides a single entry point for training, 
resuming and evaluating pre-trained models. 

## Training a new model:

- Run `python run_cqa.py --expt_name EXPT_NAME --data_root DATA_DIR` to train a model according to configuration defined in
 config_EXPT_NAME. The config used for DVQA and FigureQA are already provided. 
 (Optional: `--data_root DATA_ROOT` can be used if your data folder is not in `/data`)
- This creates a folder in `DATA_ROOT/experiments/EXPT_NAME` which will stores a copy of the config
 used and all the model predictions and model snapshots

- (Optional) Different model/data variants can be trained by creating new config files. 
You can start by copying config_template.py to config_<YOUR_EXPT_NAME>.py and make the necessary changes.


## Resuming training

- Run `run_cqa.py --expt_name EXPT_NAME --resume` to resume model training from the latest snapshot saved in
from the earlier run of the experiment saved in `DATA_ROOT/experiments/EXPT_NAME`

## Evaluating previously trained model

- Run `run_cqa.py --expt_name EXPT_NAME --resume` to evaluate latest snapshot saved in
from the previously trained model `DATA_ROOT/experiments/EXPT_NAME`

## Computing Detailed Metrics:

- Run `compute_metrics.py --expt_name EXPT_NAME` to compute accuracy for each question and image-type. (Optional: Use `--epoch EPOCH` to 
 compute metrics for a different epoch than the latest one.)

# Citation

If you use PReFIL, or the code in this repo, please cite:
``` 
@inproceedings{kafle2020answering,
  title={Answering questions about data visualizations using efficient bimodal fusion},
  author={Kafle, Kushal and Shrestha, Robik and Cohen, Scott and Price, Brian and Kanan, Christopher},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1498--1507},
  year={2020}
}
```

If you use DVQA in your work, please cite:

```
@inproceedings{kafle2018dvqa,
  title={DVQA: Understanding data visualizations via question answering},
  author={Kafle, Kushal and Price, Brian and Cohen, Scott and Kanan, Christopher},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5648--5656},
  year={2018}
}
```