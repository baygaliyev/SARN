import logging
import torch
import torch.nn as nn

from config import Config as Config

# nlanes_code added 
# c, b, h centrality and degree added
class FeatEmbedding(nn.Module):
    def __init__(self, nwayid_code, nsegid_code, nhighway_code, 
            nlength_code, nradian_code, nlon_code, nlat_code, nlanes_code, 
            nc_centrality_code, nb_centrality_code, nh_centrality_code, ndegree_code):
            
        super(FeatEmbedding, self).__init__()

        logging.debug('FeatEmbedding args. {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format( \
                        nwayid_code, nsegid_code, nhighway_code, nlength_code, \
                        nradian_code, nlon_code, nlat_code, nlanes_code, \
                        nc_centrality_code, nb_centrality_code, nh_centrality_code, ndegree_code))
        
        self.emb_highway = nn.Embedding(nhighway_code, Config.sarn_seg_feat_highwaycode_dim) #16
        self.emb_length = nn.Embedding(nlength_code, Config.sarn_seg_feat_lengthcode_dim)    #16
        self.emb_radian = nn.Embedding(nradian_code, Config.sarn_seg_feat_radiancode_dim)    #16
        self.emb_lon = nn.Embedding(nlon_code, Config.sarn_seg_feat_lonlatcode_dim)          #32
        self.emb_lat = nn.Embedding(nlat_code, Config.sarn_seg_feat_lonlatcode_dim)          #32
        self.emb_lanes = nn.Embedding(nlanes_code, Config.sarn_seg_feat_lencode_dim)         #16, you can also directly write the number
        print(nradian_code)
        print(nc_centrality_code)
        self.emb_c_centrality = nn.Embedding(nc_centrality_code, Config.sarn_seg_feat_ccentralitycode_dim)  #16, new
        self.emb_b_centrality = nn.Embedding(nb_centrality_code, Config.sarn_seg_feat_bcentralitycode_dim)  #16, new
        self.emb_h_centrality = nn.Embedding(nh_centrality_code, Config.sarn_seg_feat_hcentralitycode_dim)  #16, new
        self.emb_degree = nn.Embedding(ndegree_code, Config.sarn_seg_feat_degree_dim)                 #16, new
                
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
                self.emb_lanes(inputs[: , 9]),
                self.emb_c_centrality(inputs[: , 10]),
                self.emb_b_centrality(inputs[: , 11]),
                self.emb_h_centrality(inputs[: , 12]),
                self.emb_degree(inputs[: , 13])), dim = 1)    # last rows added by me
