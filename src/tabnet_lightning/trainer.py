import types
from argparse import ArgumentParser, Namespace
from typing import List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


class TabNetTrainer(Trainer):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = Trainer.add_argparse_args(parent_parser, **kwargs)

        parser = parent_parser.add_argument_group("TabNetTrainer")

        # to be able to overwrite pl default trainer args
        for _ag in parser._action_groups:
            _ag.conflict_handler = "resolve"

        # workaround for problematic tpu_cores arg
        parser.add_argument('--tpu_cores', type=int, default=None)
        parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
        parser.add_argument("--accelerator", type=str,
                            default=("ddp" if torch.cuda.device_count() > 1 else None))
        # parser.add_argument("--precision", default=32, type=int)

        # parser.add_argument("--log_every_n_steps", default=1, type=int)
        # parser.add_argument("--flush_logs_every_n_steps", default=50, type=int)

        return parent_parser

    @rank_zero_only
    def log_hyperparameters(self, logger: LightningLoggerBase, ignore_param: List[str] = None, types: List = None):
        if types is None:
            types = [int, float, str, dict, list, bool, tuple]

        if ignore_param is None:
            ignore_param = ["callbacks"]

        params = {}
        for k, v in self.__dict__.items():
            if k not in ignore_param and not k.startswith("_"):
                if type(v) in types:
                    params[k] = v

        params["deterministic"] = self.accelerator_connector.deterministic
        params["precision"] = self.accelerator_connector.precision
        params["num_gpus"] = self.accelerator_connector.num_gpus

        params = Namespace(**params)

        logger.log_hyperparams(params)


