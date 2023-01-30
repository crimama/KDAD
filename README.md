# Reverse distillation 

This is unofficial of Anomaly Detection via Reverse Distillation from One-Class Embedding

Paper : https://arxiv.org/abs/2201.10703

# Dataset 
- MVtecAD Dataset : https://www.mvtec.com/company/research/datasets/mvtec-ad
- ViSA Dataset : https://github.com/amazon-science/spot-diff

- Since I use two difference dataset, so I unify structure of dataset. So, before running training, you need to make a split_csv file using [make_csv4mvtecad.ipynb](make_csv4mvtecad.ipynb) ,which is to make csv file containing information about dataset like what `ViSA` dataset has. 

# Run 

**single - gpu**
```
python Reversedistillation.py --yaml_config ./configs/mvtec.yaml
```

**Multi - GPU** 
```
accelerate config 
accelerate launch Reversedistillation.py --yaml_config ./configs/mvtec.yaml
```

- Before start training using multi-gpu, You need to set accelerate config so that Accelerate can use multi-gpu 
  

# Instruction 

Notion : https://www.notion.so/hunim/Reverse-distillation-f7a79fc9d07a4d8d8bb3d4a914465e7a