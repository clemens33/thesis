#%%

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetClassifier

#%%

seed_everything(1)
path = "../../../"

#%%

dm = MolNetClassifierDataModule(
    name="bbbp",
    batch_size=64,
    radius=3,
    n_bits=4096,
    chirality=True,
    features=True,
    split="random",
    num_workers=8,
    cache_dir=path + "data/molnet/bbbp/"
)

dm.prepare_data()
dm.setup()

#%%

classifier = TabNetClassifier(
    input_size=dm.input_size,
    feature_size=512,
    decision_size=256,
    num_classes=len(dm.classes),
    nr_layers=2,
    nr_shared_layers=2,
    nr_steps=5,
    gamma=2,

    # pytorch batch norm uses 1 - momentum (e.g. 0.3 pytorch is the same as tf 0.7)
    momentum=0.1,
    virtual_batch_size=8,

    lambda_sparse=0.0001,

    # define embeddings for categorical variables - otherwise raw value is used
    #categorical_indices=list(range(dm.input_size)),
    #categorical_size=[2] * dm.input_size,
    #embedding_dims=[1] * dm.input_size,

    lr=0.0001,
    optimizer="adam",
    #scheduler="exponential_decay",
    #scheduler_params={"decay_step": 500, "decay_rate": 0.95},

    #optimizer="adamw",
    #optimizer_params={"weight_decay": 0.0001},
    scheduler="linear_with_warmup",
    scheduler_params={"warmup_steps": 0.1},

    class_weights=dm.class_weights,
)

#%%

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = Trainer(default_root_dir=path + "logs/molnet/bbbp/",

                  gpus=1,
                  # accelerator="none",

                  max_steps=4000,
                  check_val_every_n_epoch=10,
                  num_sanity_val_steps=-1,

                  fast_dev_run=False,
                  deterministic=True,
                  #precision=16,

                  #gradient_clip_algorithm="value",
                  #gradient_clip_val=2,

                  callbacks=[lr_monitor])

#%%

trainer.fit(classifier, dm)


