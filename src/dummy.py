from typing import Optional, List

import torch
import torch.nn as nn

torch.manual_seed(1)


def _init_categorical_embeddings(categorical_size: Optional[List[int]] = None,
                                 embedding_dims: Optional[List[int]] = None) -> nn.ModuleList:
    assert len(categorical_size) == len(
        embedding_dims), f"categorical_size length {len(categorical_size)} must be the same as embedding_dims length {len(embedding_dims)}"

    return nn.ModuleList([
        nn.Embedding(num_embeddings=size, embedding_dim=dim) for size, dim in zip(categorical_size, embedding_dims)
    ])


C = 2
S = 2
D = 1

categorical_size = [C] * S
embedding_dims = [D] * S

inputs = torch.randint(2, size=(4, S))

embeddings = _init_categorical_embeddings(categorical_size, embedding_dims)

outputs = []
for i, embedding in enumerate(embeddings):
    input = inputs[:, i]

    output = embedding(input)
    outputs.append(output)

output = torch.cat(outputs, dim=1)

# embedding_bag = nn.EmbeddingBag(num_embeddings=C, embedding_dim=D, mode="sum")
# offsets = list(range(0, S, 1))

# output2 = embedding_bag(inputs, offsets)


embedding_weights = nn.Parameter(torch.Tensor(C, D * S))
nn.init.normal_(embedding_weights)
print(embedding_weights)

output = torch.embedding(embedding_weights, inputs.long())

print("the end")
