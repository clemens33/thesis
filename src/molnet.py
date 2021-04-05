#%%
from pytorch_lightning import Trainer

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetClassifier

#%%

dm = MolNetClassifierDataModule(
    name="bbbp",
    batch_size=32,
    radius=3,
    n_bits=2048
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
    nr_steps=4,
    gamma=1.0
)

print(f"number of parameters for classifier: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}")

#%%

trainer = Trainer(gpus=1, max_steps=10)
trainer.fit(classifier, dm)
