import torch
import torch.nn as nn
import einops
from dataclasses import dataclass

@dataclass 
class ImapEmbeddingConfig:
	imap_size: int = 3
	hidden_size: int = 512
	dropout_rate: float = 0.1

# https://github.com/cshizhe/onav_rim/blob/main/offline_bc/models/onav_imap_models.py
class ImapEmbedding(nn.Module):
    def __init__(self, model_config: ImapEmbeddingConfig) -> None: # model_config contains imap_size, hidden_size
        super().__init__()
        self.imap_size = model_config.imap_size
        # imap consists of imap_size x imap_size embeddings of size hidden_size
        
        # lookup table
        self.imap_token_embedding = nn.Embedding(self.imap_size**2, model_config.hidden_size)

        # tensor of 2D coordinates for imap_size x imap_size square centered around (0,0) (flattened into a single vector of coordinates so we can pass
        # it into the fully connected network)
        self.imap_pos_fts = self._create_imap_pos_features(self.imap_size) 

        # FFN from equation 1
        self.imap_pos_layer = nn.Sequential(
            nn.Linear(2, model_config.hidden_size), # (location: x, y)
            nn.LayerNorm(model_config.hidden_size)
        )
            
        self.ft_fusion_layer = nn.Sequential(
            nn.LayerNorm(model_config.hidden_size),
            nn.Dropout(model_config.dropout_rate),
        )

    def _create_imap_pos_features(self, imap_size):
        x, y = torch.meshgrid(torch.arange(imap_size), torch.arange(imap_size))
        xy = torch.stack([x, y], dim=2)
        xy = (xy + 0.5 - imap_size / 2).float() # relative distance to the center
        xy = xy.view(-1, 2)
        return xy

    def forward(self, batch_size):
        '''Get the initialized imap embedding'''
        device = self.imap_token_embedding.weight.device

        # initialize embedding for each element in the imap using the lookup table
        token_types = torch.arange(self.imap_size**2, dtype=torch.long, device=device)
        embeds = self.imap_token_embedding(token_types)

        # create embedding to capture spatial relationship between position within the imap using equation 1
        pos_embeds = self.imap_pos_layer(self.imap_pos_fts.to(device))

        embeds = embeds + pos_embeds
        embeds = self.ft_fusion_layer(embeds)

        # (imap_size**2, hidden_size) -> (batch_size, imap_size**2, hidden_size)
        embeds = einops.repeat(embeds, 'n d -> b n d', b=batch_size) 
        pos_embeds = einops.repeat(pos_embeds, 'n d -> b n d', b=batch_size)

        return embeds, pos_embeds
