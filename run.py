from argparse import ArgumentParser

from datamodules.cs8k_datamodule import *
from datamodules.m2caiseg_datamodule import *
from datamodules.bcv30_datamodule import *
from model.unet import ResNetUNet
from queries.query import *
from queries.random import *
from queries.uncertainty import *
from queries.coreset import *
from queries.deal import *
from queries.alges import *

import copy
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger 

import logging
logging.getLogger("pytorch_lightning").setLevel(50)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Disable validation progress bar
class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        return tqdm(disable=True)

    
def main(hparams):
    prog_bar = LitProgressBar()
    
    logger = None
    if hparams.logger:
        logger = WandbLogger(project=hparams.project)
    
    # Setup data for active learning
    n_class = ...
    if hparams.dataset == 'cholecseg8k':
        dm = CS8KDataModule(hparams)
        n_class = 13
    elif hparams.dataset == 'm2caiseg':
        dm = M2CAISEGDataModule(hparams)
        n_class = 19
    elif hparams.dataset == 'bcv30':
        dm = BCV30DataModule(hparams)
        n_class = 13
    dm.prepare_data()
    dm.setup()
    
    # Get initial labeled sample pool
    init_lb_pool = np.full(len(dm.X_tr), False, dtype=bool)
    tmp = np.arange(len(dm.X_tr))
    np.random.shuffle(tmp)
    init_lb_pool[tmp[0:hparams.n_init]] = True

    # Save accuracies on val for each round 
    num_metrics = ...
    if hparams.dataset == 'cholecseg8k':
        num_metrics = 8
    else:
        num_metrics = n_class
    val_class_dsc = np.zeros((hparams.n_rounds, hparams.n_exp, num_metrics))
    test_class_dsc = np.zeros((hparams.n_rounds, hparams.n_exp, num_metrics))
    
    # Setup AL querying method
    query_types = [
        Naive, 
        Random, 
        MaxEntropy, Margins, LeastConf, 
        CoreSet, 
        DEAL, 
        ALGESv1, ALGESv2,
    ]
    strat = query_types[hparams.query]
    seed = [hparams.seed+i for i in range(0, hparams.n_exp)]
    
    # ACTIVE LEARNING -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    for exp in range(0, hparams.n_exp):
        
        # set random seed
        np.random.seed(seed[exp])
        pl.seed_everything(seed[exp])

        lb_pool = init_lb_pool.copy() # intialize training pool
        model = ResNetUNet(hparams, n_class) # initialize new model with diff init weights for diff experiments
        
        for r in range(0, hparams.n_rounds): 
            print(f'[Exp {exp + 1}: round {r} -> lb_pool = {sum(lb_pool)}, unlb_pool = {sum(~lb_pool)}]')
            dm.update_pool(inc_idxs=lb_pool)

            # configure callbacks
            # ckpt_callback = ModelCheckpoint(monitor='val_dice', mode='max', dirpath=f'checkpoints/{hparams.query}')
            # es_callback = EarlyStopping(monitor='train_loss', patience=3, mode='min')
            
            # Setup new trainer for new round of training
            trainer = Trainer(gpus=hparams.gpus, 
                              # callbacks=[prog_bar, ckpt_callback],
                              # callbacks=[prog_bar, es_callback],
                              callbacks=[prog_bar],
                              deterministic=hparams.deterministic, 
                              logger=logger,
                              min_epochs=hparams.min_epochs,
                              max_epochs=hparams.max_epochs,
                              enable_model_summary = False,
                              log_every_n_steps=1,
                              # check_val_every_n_epoch=5,
                         )
            new_model = copy.deepcopy(model) # retrain model from scratch
    
            # Retrain from scratch
            # trainer.fit(new_model, dm.train_dataloader(), dm.val_dataloader())
            trainer.fit(new_model, dm.train_dataloader())
            
            if hparams.eval:
                T = trainer.test(new_model, dm.test_dataloader(), ckpt_path=None)[0]
                for i in range(num_metrics):
                    test_class_dsc[r, exp, i] = T[f'tdsc-c{i}']
                np.save(f'logs/{hparams.dataset}/{hparams.seed}_{hparams.dataset}_te_ce_r{hparams.n_rounds}_i{hparams.n_init}_q{hparams.n_query}_s{hparams.query}', 
                        test_class_dsc)
            else:
                V = trainer.validate(new_model, dm.val_dataloader(), ckpt_path='best')[0]
                for i in range(num_metrics):
                    val_class_dsc[r, exp, i] = V[f'vdsc-c{i}']
                np.save(f'logs/{hparams.dataset}/{hparams.seed}_{hparams.dataset}_val_ce_r{hparams.n_rounds}_i{hparams.n_init}_q{hparams.n_query}_s{hparams.query}', 
                        val_class_dsc)

            # Get new samples to include in labeled pool
            if hparams.n_rounds > 1:
                sampler = strat(lb_pool, new_model, dm)
                lb_pool[sampler.query(hparams.n_query)] = True
            
            # Clean up model and callbacks after each round of training 
            del new_model
            del trainer
            del ckpt_callback
            
        del model
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    print('Active learning experiments complete.')


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Hardware args
    parser.add_argument("--gpus", default=1, type=int,
                       help='number of gpus for training')
    parser.add_argument("--deterministic", default=False, type=bool,
                       help='to enforce deterministic output, set to true')
    parser.add_argument("--seed", default=3507, type=int,
                       help='for reproducibility, set seed')
    
    # Pytorch-lightning args
    parser.add_argument("--logger", default=0, type=int, choices=[0, 1],
                       help='option for logging with wandb')
    parser.add_argument("--project", default='', type=str,
                       help='name of the wandb project to log metrics to')
    
    # Data args
    parser.add_argument("--dataset", default='cholecseg8k', type=str, choices=['cholecseg8k', 'm2caiseg', 'bcv30'],
                       help='which dataset to use for active learning')
    
    # Learning args
    parser.add_argument("--min_epochs", default=25, type=int,
                       help='minimum number of training epochs (applies to every round of AL)')
    parser.add_argument("--max_epochs", default=50, type=int,
                       help='maximum number of training epochs (applies to every round of AL)')
    parser.add_argument("--lr", default=5e-5, type=float,
                       help='learning rate')
    parser.add_argument("--batch_size", default=32, type=int,
                       help='training batch size')
    parser.add_argument("--val_size", default=1600, type=int,
                       help='size of validation set')
    parser.add_argument("--test_size", default=1840, type=int,
                       help='size of validation set')
    parser.add_argument("--record", default=True, type=bool,
                       help='save validation/test mean dice scores every round of AL')
    parser.add_argument("--eval", default=1, type=int, choices=[0, 1],
                       help='Evaluate model on test (1) or val (0)')
    
    # Active learning args
    parser.add_argument("--n_rounds", default=50, type=int,
                       help='number of active learning rounds')
    parser.add_argument("--n_init", default=10, type=int,
                       help='number of initial labeled samples')
    parser.add_argument("--n_query", default=10, type=int,
                       help='number of samples to query for oracle')
    parser.add_argument("-q", "--query", default=1, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                       help='AL querying strategy: 0 = Naive, 1 = Random, \
                       2 = Max entropy, 3 = Margins sampling, 4 = Least confidence, \
                       5 = Coreset, 6 = DEAL, 7 = ALGESv1, 8 = ALGESv2')
    parser.add_argument("--n_exp", default=1, type=int,
                       help='number of complete AL experiments to run')
    
    args = parser.parse_args()
    main(args)