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

## ToDo
- Provide data on the usage of adversarial attacks.


https://drive.google.com/file/d/1iorbzFc5XYUId9gsZlFVcGXszhFEhHgO/view?usp=sharing - model weights
https://drive.google.com/file/d/1UMFOw4YGoxqruOgIvqBD7ffpEwscM6fh/view?usp=sharing - UAP weights
