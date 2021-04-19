
#%%

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from datasets import CovTypeDataModule
from tabnet_lightning import TabNetClassifier

#%%

seed_everything(1)
path = "../../"

#%%

dm = CovTypeDataModule(
    batch_size=16384,
    num_workers=8,
    cache_dir=path + "data/uci/covtype/",
    seed=0,  # use same seed / random state as tabnet original implementation
)

dm.prepare_data()
dm.setup()

#%%

classifier = TabNetClassifier(
    input_size=CovTypeDataModule.NUM_FEATURES,
    feature_size=128,
    decision_size=64,
    num_classes=CovTypeDataModule.NUM_LABELS,
    nr_layers=2,
    nr_shared_layers=2,
    nr_steps=5,
    gamma=1.5,

    # pytorch batch norm uses 1 - momentum (e.g. 0.3 pytorch is the same as tf 0.7)
    momentum=0.3,
    virtual_batch_size=512,

    lambda_sparse=0.0001,

    # define embeddings for categorical variables - otherwise raw value is used
    # categorical_indices=list(range(len(CovTypeDataModule.NUMERICAL_COLUMNS), CovTypeDataModule.NUM_FEATURES, 1)),
    # categorical_size=[2] * len(CovTypeDataModule.BINARY_COLUMNS),
    # embedding_dims=[1] * len(CovTypeDataModule.BINARY_COLUMNS),

    lr=0.02,
    optimizer="adam",
    scheduler="exponential_decay",
    scheduler_params={"decay_step": 500, "decay_rate": 0.95},

    #optimizer="adamw",
    #optimizer_params={"weight_decay": 0.0001},
    #scheduler="linear_with_warmup",
    #scheduler_params={"warmup_steps": 0.1},
)

#%%

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = Trainer(default_root_dir=path + "logs/covtype/",

                  gpus=1,
                  # accelerator="none",

                  max_steps=2000,

                  fast_dev_run=False,
                  deterministic=True,

                  gradient_clip_algorithm="value",
                  gradient_clip_val=2000,

                  callbacks=[lr_monitor])

#%%

trainer.fit(classifier, dm)

#%%

trainer.test(classifier, datamodule=dm)
