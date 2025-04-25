import os
import time
import pickle
import argparse
import logging

import torch
import pytorch_lightning as pl

pl.seed_everything(42)


from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup

import model as Model

from utils.datamodule import DataModule
from utils.result import Result
from utils import params_count
from utils.visualize import *

logger = logging.getLogger(__name__)
torch.backends.cudnn.enable = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.loss_dict = {}

        # get data info
        user_num, item_num, active_user_list, static_user_list = data_module.get_num_info()

        # for fair:
        dis_half2half = data_module.get_dis_half()
        static2active_v2 = data_module.raw_datasets['static2active_v2']

        if self.hparams.arch in ("NGCF", "LightGCN"):
            inter_user_list, inter_item_list = data_module.get_edge_info()
            self.model = Model.__dict__[self.hparams.arch](user_num, 
                                                           item_num, 
                                                           active_user_list, 
                                                           static_user_list,
                                                           dis_half2half, 
                                                           self.hparams.mode,
                                                           inter_user_list, 
                                                           inter_item_list,
                                                           static2active_v2,
                                                       )
        else:
 
            self.model = Model.__dict__[self.hparams.arch](user_num, 
                                                           item_num, 
                                                           active_user_list, 
                                                           static_user_list,
                                                           dis_half2half, 
                                                           self.hparams.mode,
                                                           static2active_v2,)

        dir_name = os.path.join(self.hparams.output_dir,
                                f'dataset={self.hparams.dataset1},{self.hparams.dataset2},{self.hparams.dataset3},seed={self.hparams.seed},m={self.hparams.output_sub_dir}, a={self.hparams.arch}',
                                'tensorboard')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.iter = 0
        self.best_val_result = {}

        print(self.model.config)
        print('---------------------------------------------')
        print('total params_count:', params_count(self.model))
        print('---------------------------------------------')

    @pl.utilities.rank_zero_only
    def save_model(self, name):
        dir_name = os.path.join(self.hparams.output_dir,
                                f'dataset={self.hparams.dataset1},{self.hparams.dataset2},{self.hparams.dataset3},seed={self.hparams.seed},m={self.hparams.output_sub_dir}, a={self.hparams.arch}, mode={self.hparams.mode}',
                                'model')
        print(f'## save model to {dir_name}')
        self.model.config['time'] = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = os.path.join(dir_name, '{}_model.pt'.format(name))
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, name):
        dir_name = os.path.join(self.hparams.output_dir,
                                f'dataset={self.hparams.dataset1},{self.hparams.dataset2},{self.hparams.dataset3},seed={self.hparams.seed},m={self.hparams.output_sub_dir}, a={self.hparams.arch}, mode={self.hparams.mode}',
                                'model', '{}_model.pt'.format(name))
        print(f'## load model to {dir_name}')
        state_dict = torch.load(dir_name)
        self.model.load_state_dict(state_dict)
        self.model.config['metric'] = name

    def forward(self, **inputs):
        id_start_end = inputs.pop('id')
        isActive = inputs['is_active']
        outputs = self.model(**inputs)
        outputs['is_active'] = isActive
        outputs['uid'] = inputs['user_indices']
        outputs['iid'] = inputs['item_indices']
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        if self.hparams.mode in ('inter',):
            outputs['loss'] = outputs['original_loss'] + outputs['OT_loss_inter']* self.hparams.l2
        if self.hparams.mode in ('fair', ) and not outputs['is_active']:
            outputs['loss'] = outputs['original_loss'] + outputs['OT_loss_inter'] * self.hparams.l2+ outputs['OT_loss_intra']

        self.log("batch loss", outputs['loss'])
        
        return outputs

    def training_epoch_end(self, outputs):
        original_losses = []
        ot_intra_losses = []
        ot_inter_losses = []
        overall_losses = []
        for output in outputs:
            original_losses.append(output['original_loss'].item())
            ot_intra_losses.append(output['OT_loss_intra'].item())
            ot_inter_losses.append(output['OT_loss_inter'].item())
            overall_losses.append(output['loss'].item())
        original_loss = sum(original_losses) / len(overall_losses)
        ot_loss_inter = sum(ot_inter_losses) / len(ot_inter_losses)
        ot_loss_intra = sum(ot_intra_losses) / len(ot_intra_losses)
        overall_loss = sum(overall_losses) / len(overall_losses)
        
        print(f"\nOriginal loss: {original_loss}, OT loss: inter{ot_loss_inter} intra{ot_loss_intra}, overall loss:{overall_loss}")
        self.log("loss/original_loss", original_loss)
        self.log("loss/ot_loss_inter", ot_loss_inter)
        self.log("loss/ot_loss_intra", ot_loss_intra)
        self.log("loss/overall_loss", overall_loss)

        self.loss_dict[self.iter] = (overall_loss, original_loss, ot_loss_inter, ot_loss_intra)
        self.iter += 1


    def save_loss(self):
        # for drawing loss trend figure
        dir_name = os.path.join(self.hparams.output_dir,
                                f'dataset={self.hparams.dataset1},{self.hparams.dataset2},{self.hparams.dataset3},seed={self.hparams.seed},m={self.hparams.output_sub_dir}, a={self.hparams.arch}, mode={self.hparams.mode}',
                                'loss')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        pickle.dump(self.loss_dict, open(os.path.join(dir_name, "loss.pkl"), 'wb'))

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']

        self.log('valid_loss', loss)
        return outputs

    def validation_epoch_end(self, outputs):
        self.current_val_result = Result.parse_from(outputs, self.hparams.sample_per_user, seperate_active=True)
        cur_res = self.current_val_result.cal_metric()

        # save best metrics
        for k, v in cur_res.items():
            if not k in self.best_val_result:
                self.best_val_result[k] = (v, self.current_val_result)
                self.save_model(k)
            elif self.best_val_result[k][0] < v:
                self.best_val_result[k] = (v, self.current_val_result)
                self.save_model(k)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.test_result = Result.parse_from(outputs, self.hparams.sample_per_user, seperate_active=True)
        self.test_result.cal_metric()


    def setup(self, stage):
        if stage == 'fit':
            # set up optimizer configurations
            self.train_loader = self.train_dataloader()
            self.train_batch_num = len(self.train_loader)
            ngpus = (len(self.hparams.gpus.split(',')) if type(self.hparams.gpus) is str else self.hparams.gpus)
            # effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * ngpus
            effective_batch_size = self.hparams.accumulate_grad_batches * ngpus
            dataset_size = len(self.train_loader.dataset)
            self.total_steps = (self.train_batch_num / effective_batch_size) * self.hparams.max_epochs
            self.warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)
            print(f'dataset_size:{dataset_size}, total_steps: {self.total_steps}')

    def configure_optimizers(self):
        # set up optimizer configurations
        print("learning rate:", self.hparams.learning_rate)

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = Adam(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--dataset1", default="dataset_small", type=str)
        parser.add_argument("--dataset2", default="Epinion", type=str)
        parser.add_argument("--dataset3", default="Epinion", type=str)
        parser.add_argument("--data_dir", default="datasets/", type=str)
        parser.add_argument("--output_dir", default="outputs/", type=str)
        parser.add_argument("--output_sub_dir", default="debug", type=str)
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_ratio", default=0, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--seed", default=10, type=int)
        parser.add_argument("--do_train", default=False, action='store_true')
        parser.add_argument("--sample_per_user", default=100, type=int)

        parser.add_argument("--arch", default="NeuMF", type=str, choices=["NeuMF","MF","LightGCN"])

        # for ot
        parser.add_argument("--l2", type=float, default=500)
        parser.add_argument("--mode", type=str, default="fair", choices=["original", "fair", "intra", "inter"])

        return parser
    
    


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        # print validation result
        print(
            '-------------------------------------------------------------------------------------------------------------------\n[current]\t',
            end='')
        pl_module.current_val_result.report()

        print('[best]\t\t', end='')
        for key in pl_module.best_val_result:
            print(f'the best result for metric {key} is:')
            pl_module.best_val_result[key][1].report()
        print(
            '-------------------------------------------------------------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.save_loss()

        pl_module.test_result.report()
        pl_module.test_result.save_metric(
            output_dir=pl_module.hparams.output_dir,
            subname=pl_module.hparams.output_sub_dir,
            dataset=pl_module.hparams.dataset1 + "-" + pl_module.hparams.dataset2 + "-" + pl_module.hparams.dataset3,
            seed=pl_module.hparams.seed,
            arch=pl_module.hparams.arch,
            mode=pl_module.hparams.mode,
            active_percent=pl_module.hparams.active_percent,
        )
        if pl_module.hparams.mode == 'original':
            # save prediction results to test on UOF
            _, _, active_user_list, static_user_list = pl_module.data_module.get_num_info()
            pl_module.test_result.save_for_uof(
                active_user_list=active_user_list, 
                static_user_list=static_user_list, 
                dataset=pl_module.hparams.dataset1 + "-" + pl_module.hparams.dataset2 + "-" + pl_module.hparams.dataset3, 
                arch=pl_module.hparams.arch, 
                path=pl_module.hparams.output_dir,
            )

        # save tsne results for visualization
        # if pl_module.hparams.arch == 'LightGCN':
        #     save_dir = os.path.join(pl_module.hparams.output_dir, f'dataset={pl_module.hparams.dataset1},{pl_module.hparams.dataset2},{pl_module.hparams.dataset3},seed={pl_module.hparams.seed},m={pl_module.hparams.output_sub_dir}, a={pl_module.hparams.arch}, mode={pl_module.hparams.mode}',
        #                             't-sne')
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     save_tsne(pl_module.model, save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    if args.seed != 0:
        pl.seed_everything(args.seed)
    else:
        pl.seed_everything()

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5
    if args.l2 >= 1:
        args.l2 /= 1e5

    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()
    data_module.prepare_dataset()

    model = LightningModule(args, data_module)

    logging_callback = LoggingCallback()
    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        'gpus': "0",
        'num_sanity_val_steps': 0  # 5 if args.do_train else 0,
    }

    trainer = pl.Trainer.from_argparse_args(args, **kwargs, fast_dev_run=False)

    metrics = ['hit_ratio@10(all)', 'ndcg@10(all)']

    if args.do_train:
        trainer.fit(model, datamodule=data_module)
        for k in metrics:
            print(f'best result for metic:{k}')
            model.load_model(k)
            trainer.test(model, datamodule=data_module)

    else:
        for k in metrics:
            print(f'best result for metic:{k}')
            model.load_model(k)
            trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
