import torch
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, RobertaModel
from setting.special_tokens import SPECIAL_TOKENS


class RetrieverRoBERTaEncoder(LightningModule):
    """Encodes the input as embeddings using RoBERTa."""

    def __init__(self, architecture, output_attentions, output_hidden_states, pooling):
        super(RetrieverRoBERTaEncoder, self).__init__()

        # tokenizer = AutoTokenizer.from_pretrained(architecture)

        # tokenizer.add_special_tokens({
        #     "additional_special_tokens": SPECIAL_TOKENS
        # })

        self.encoder = RobertaModel.from_pretrained(
            architecture,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_safetensors=False
        )

        # self.encoder.resize_token_embeddings(len(tokenizer))

        self.pooling = pooling

    def forward(self, feature):
        # Set attention mask for non-padding tokens (padding token ID is 1 for RoBERTa)
        attention_mask = torch.where(feature != 1, 1, 0)
        encoder_outputs = self.encoder(feature, attention_mask=attention_mask)
        return self.pooling(
            encoder_outputs,
            attention_mask
        )