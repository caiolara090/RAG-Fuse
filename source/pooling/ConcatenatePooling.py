import torch
from pytorch_lightning import LightningModule


class ConcatenatePooling(LightningModule):
    """
    Performs concatenate pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(ConcatenatePooling, self).__init__()

    def forward(self, encoder_outputs, attention_mask=None):
        hidden_states = encoder_outputs.hidden_states
        concatenate_pooling = torch.cat(
            (
                hidden_states[-1],
                hidden_states[-2],
                hidden_states[-3],
                hidden_states[-4]),
            -1
        )
        #return concatenate_pooling[:, 0]
        #return torch.nn.functional.normalize(concatenate_pooling[:, 0], p=2, dim=1)
    
        mask = attention_mask.unsqueeze(-1).expand(concatenate_pooling.size()).float()

        masked_embeddings = concatenate_pooling * mask

        summed = torch.sum(masked_embeddings, dim=1)

        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        mean_pooled = summed / summed_mask

        return torch.nn.functional.normalize(
            mean_pooled,
            p=2,
            dim=1
        )
