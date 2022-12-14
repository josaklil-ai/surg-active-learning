# ALGES: Active Learning with Gradient Embeddings for Semantic Segmentation of Laparoscopic Surgical Images

This is the official implementation for the paper *[Josiah Aklilu, Serena Yeung, "ALGES: Active Learning with Gradient Embeddings for Semantic Segmentation of Laparoscopic Surgical Images"](https://www.mlforhc.org/2022-accepted-papers)*. 

### Run

To obtain the fully supervised model peformance on the held out test set for a particular dataset (e.g. 'cholecseg9k'), run 1 round of AL with the full size of the training set:

`python run.py --n_init 4640 --n_rounds 1 --n_exp 1`

To run the experiments from the paper, run the following for each dataset with a specific AL query strategy (e.g. 7 = ALGES-img or 8 = ALGES-seg):

For cholecSeg8k
- `python run.py --dataset 'cholecseg8k' --test_size 1640 --n_init 50 --n_query 10 --n_rounds 30 --n_exp 3 --query 7`

For m2caiSeg
- `python run.py --dataset 'm2caiseg' --val_size 31 --n_init 10 --n_query 4 --n_rounds 51 --n_exp 3 --query 8`

|  | Query method |
| --- | --- |
| 1 | Random |
| 2 | Max entropy sampling |
| 3 | Margins sampling |
| 4 | Least confidence sampling |
| 5 | Coreset |
| 6 | DEAL |
| 7 | ALGES-img (ours) |
| 8 | ALGES-seg (ours) |

## References
Some code adopted from the [DeepAL](https://github.com/ej0cl6/deep-active-learning) and [ViewAL](https://github.com/nihalsid/ViewAL) repos. 

1. Fu J, Liu J, Tian H, et al. [Dual attention network for scene segmentation](https://arxiv.org/pdf/1809.02983.pdf) // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 3146-3154.
2. Xie S, Feng Z, Chen Y, et al. [DEAL: Difficulty-aware Active Learning for Semantic Segmentation](https://arxiv.org/pdf/2010.08705.pdf) //
