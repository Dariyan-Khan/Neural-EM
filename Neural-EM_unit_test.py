import pytest
from nem import add_noise
import torch



@pytest.mark.parametrize('noise_type, prob, binary, data, expected', [
    ('None', None, False, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])),
    ('data', None, True, torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])),
    ('bitflip', 0.2, False, torch.tensor([1, 0, 1]), torch.tensor([1, 0, 1])),
    ('masked_uniform', 0.5, True, torch.tensor([1, 0, 1]), torch.tensor([0.5, 0, 0.5])),
])
def test_add_noise(noise_type, prob, binary, data, expected):
    noise = {'noise_type': noise_type, 'prob': prob}
    dataset = {'binary': binary}
    
    result = add_noise(data, noise, dataset)
    
    assert torch.allclose(result, expected)

