import torch

import torch.nn.functional as F

from scbasset.scbasset import scBasset


def test_scbasset_forward_backward():
    """Test forward & backward pass of scBasset model."""
    # create dummy data
    batch_size = 10
    genomic_seq_length = 600
    n_cells = 100
    x = (
        F.one_hot(
            torch.randint(0, 4, (batch_size, genomic_seq_length)), num_classes=4
        )
        .transpose(-2, -1)
        .type(torch.float32)
    )

    scbasset = scBasset(n_cells=n_cells, genomic_seq_length=genomic_seq_length)
    out = scbasset(x)
    assert out.shape == (batch_size, n_cells)

    labels = torch.empty(batch_size, n_cells).random_(2)
    loss = F.binary_cross_entropy_with_logits(out, labels)
    # just to see it runs fine
    loss.backward()
    assert loss > 0
