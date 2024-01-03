import logging
import torch
import torch.nn as nn

from config import Config as Config

# nlanes_code added 
class FeatEmbedding(nn.Module):
    def __init__(self, nwayid_code, nsegid_code, nhighway_code, 
            nlength_code, nradian_code, nlon_code, nlat_code, nlanes_code):
            
        super(FeatEmbedding, self).__init__()

        logging.debug('FeatEmbedding args. {}, {}, {}, {}, {}, {}, {}, {}'.format( \
                        nwayid_code, nsegid_code, nhighway_code, nlength_code, \
                        nradian_code, nlon_code, nlat_code, nlanes_code))
        
        self.emb_highway = nn.Embedding(nhighway_code, Config.sarn_seg_feat_highwaycode_dim) #16
        self.emb_length = nn.Embedding(nlength_code, Config.sarn_seg_feat_lengthcode_dim)    #16
        self.emb_radian = nn.Embedding(nradian_code, Config.sarn_seg_feat_radiancode_dim)    #16
        self.emb_lon = nn.Embedding(nlon_code, Config.sarn_seg_feat_lonlatcode_dim)          #32
        self.emb_lat = nn.Embedding(nlat_code, Config.sarn_seg_feat_lonlatcode_dim)          #32
        self.emb_lanes = nn.Embedding(nlanes_code, Config.sarn_seg_feat_lencode_dim)   #16, you can also directly write the number

    # inputs = [N, nfeat]
    def forward(self, inputs):
        return torch.cat( (
                self.emb_highway(inputs[: , 2]),
                self.emb_length(inputs[: , 3]),
                self.emb_radian(inputs[: , 4]),
                self.emb_lon(inputs[: , 5]),
                self.emb_lat(inputs[: , 6]),
                self.emb_lon(inputs[: , 7]),
                self.emb_lat(inputs[: , 8]),
                self.emb_lanes(inputs[: , 9])), dim = 1)    # added by me
