# ALGES: Active Learning with Gradient Embeddings for Semantic Segmentation of Laparoscopic Surgical Images

This is the official implementation for the paper *[Josiah Aklilu, Serena Yeung, "ALGES: Active Learning with Gradient Embeddings for Semantic Segmentation of Laparoscopic Surgical Images"](google.com)*. 

### Run

To obtain the fully supervised model peformance on the held out test set for a particular dataset (e.g. 'cholecseg9k'), run 1 round of AL with the full size of the training set:

`python run.py --n_init 4640 --n_rounds 1 --n_exp 1`

To run the experiments from the paper, run the following for each dataset with a specific AL query strategy (e.g. 7 = ALGES-img or 8 = ALGES-seg):

For cholecSeg8k
- `python run.py --dataset 'cholecseg8k' --val_size 820 --n_init 50 --n_query 10 --n_rounds 30 --n_exp 3 --query 7`

For m2caiSeg
- `python run.py --dataset 'm2caiseg' --val_size 31 --n_init 10 --n_query 4 --n_rounds 51 --n_exp 3 --query 8`

| Query Method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| --- | Random | Max entropy sampling | Margins sampling | Least confidence sampling | Coreset | DEAL | ALGES-img (ours) | ALGES-seg (ours) |

## References
1. Fu J, Liu J, Tian H, et al. [Dual attention network for scene segmentation](https://arxiv.org/pdf/1809.02983.pdf) // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 3146-3154.
2. Xie S, Feng Z, Chen Y, et al. [DEAL: Difficulty-aware Active Learning for Semantic Segmentation](https://arxiv.org/pdf/2010.08705.pdf) //
