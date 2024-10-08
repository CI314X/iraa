# Image Robustness to Adversarial Attacks model


## Inference model

Use pipeline:
```
main/iraa_inference.ipynb
```

## Structure

- the /data/weights folder contains pretrained model weights trained on the MS COCO dataset (train2017).
- the data/dataset_robustness folder contains the results of the IRAA model applied to popular public datasets.
- the main/adversarial_attacks folder includes the code for the adversarial attacks used in this project.
- the main/nr_ira_metrics folder contains the code for the NR-IQA metrics utilized.

## Attacks
The code for adversarial attacks can be found at the following repository: [MSU Metrics Robustness Benchmark](https://github.com/msu-video-group/MSU_Metrics_Robustness_Benchmark]).


### External Data
 - model weights: https://drive.google.com/file/d/1iorbzFc5XYUId9gsZlFVcGXszhFEhHgO/view?usp=sharing
- UAP weights: https://drive.google.com/file/d/1UMFOw4YGoxqruOgIvqBD7ffpEwscM6fh/view?usp=sharing
