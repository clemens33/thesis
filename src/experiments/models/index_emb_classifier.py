import torch
from torch import nn

from tabnet_lightning import TabNetClassifier


class IndexEmbTabNetClassifier(TabNetClassifier):
    """implementation using index based embeddings"""

    def __init__(self, **kwargs):
        super(IndexEmbTabNetClassifier, self).__init__(**kwargs)

        self.index_embeddings = nn.Embedding(num_embeddings=kwargs["input_size"], embedding_dim=1)

    def embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = torch.nonzero(inputs, as_tuple=True)  # gets the indices which are active

        values = self.index_embeddings(indices[-1]).squeeze()

        output = torch.index_put_(inputs, indices, values)

        return output
#
# # test
# if __name__ == "__main__":
#     inputs = torch.Tensor([
#         [0, 0, 1, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1],
#     ])
#     e = nn.Embedding(num_embeddings=8, embedding_dim=1)
#
#     indices = torch.nonzero(inputs, as_tuple=True)
#
#     emb = e(indices[-1]).squeeze()
#
#     # indices[..., -1] = emb
#
#     inputs = torch.index_put_(inputs, indices, emb)
