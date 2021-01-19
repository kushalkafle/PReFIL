# Data

The question-answer files are pre-processed so that they have similar fieldnames/format for easier loading. 
We provide direct download links to download the pre-processed qa pairs. 

## Download and Setup DVQA

Run the self-contained `bash get_DVQA_data.sh` from your data root directory to download and setup images and pre-processed qa files. 
The qa files are also already encoded with appropriate OCR based on the dynamic encoding described in the paper.

NOTE: In this repo, we use the `val_` prefix to denote availability of ground truth labels. 
Hence, the names for DVQA splits are changed as follows:

- Test Familiar --> val_easy
- Test Novel --> val_hard

Optional: If you want to develop your own models using the identical OCR as us, you can [download](https://drive.google.com/file/d/1h4Lm2N_94T11JnMRureOk6jycfTWry2T/view?usp=sharing) the results of real and OCR 
for your use. These `.pkl` files provide the OCR results for each file sorted in a daisy-chain fashion as described in the paper.

## Download and Setup FigureQA

FigureQA direct download links are not available publicly. [Head over](https://www.microsoft.com/en-us/research/project/figureqa-dataset/) 
to FigureQA website to download all the `.tar.gz` files, i.e., 
`figureqa-train1-v1.tar.gz`, 
`figureqa-validation1-v1.tar.gz`, 
`figureqa-validation2-v1.tar.gz`,
`figureqa-test1-v1.tar.gz`,
`figureqa-test2-v1.tar.gz`,
 to your data root. Then run `bash get_FigureQA_data.sh` from the same directory to download the 
pre-formatted QA files and setup the folders.

## Download Pretrained Models

Run `bash get_pretrained_models.sh` from your data root directory to download and setup pretrained models for FigureQA and DVQA.
 This will create two folders inside `experiments` directory: `FigureQA_pretrained` and `DVQA_pretrained`. You can use these
 to either evaluate them or resume training from the checkpoint (I think the accuracies might still go up a bit more with 
 some more training)
 
 - `python run_cqa.py --expt_name FigureQA_pretrained --resume`
 - `python run_cqa.py --expt_name FigureQA_pretrained --evaluate`
