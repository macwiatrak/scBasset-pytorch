import torch

import torch.nn.functional as F

from scbasset.scbasset import scBasset


def test_scbasset_forward_backward():
    """Test forward & backward pass of scBasset model."""
    # create dummy data
    batch_size = 10
    seq_length = 1344
    n_cells = 100
    x = (
        F.one_hot(torch.randint(0, 4, (batch_size, seq_length)), num_classes=4)
        .transpose(-2, -1)
        .type(torch.float32)
    )

    scbasset = scBasset(n_cells=n_cells)
    out = scbasset(x)
    assert out.shape == (batch_size, n_cells)

    labels = torch.empty(batch_size, n_cells).random_(2)
    loss = F.binary_cross_entropy_with_logits(out, labels)
    # just to see it runs fine
    loss.backward()
    assert loss > 0
