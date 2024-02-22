import sys
sys.path.append('..')
import time
import logging
import pickle
from ast import literal_eval
import pandas as pd
import numpy as np
import networkx as nx
import math
import torch

from config import Config as Config
from utils import cell
sys.modules['cell'] = cell
from cell import CellSpace
from utils.tool_funcs import haversine_np 
from utils.edge_index import EdgeIndex


class OSMLoader:

    def __init__(self, osm_raw_file, schema = ''):
        self.schema = schema

        self.osm_raw_file = osm_raw_file
        self.node_file = osm_raw_file + '_node'
        self.segment_file = osm_raw_file + '_segment'
        self.way_file = osm_raw_file + '_way'
        self.adj_segment_file = osm_raw_file + '_adjsegment'
        self.cellspace_file = osm_raw_file + '_cellspace_info.pickle'
        self.segment_middistance_file = osm_raw_file + '_segment_middistance_ndarray.pickle'
        self.segment_middistance_idxmap_file = osm_raw_file + '_segment_middistance_ndarray_idxmap.pickle'

        
        self.nodes = None # DataFrame; all intersection nodes on road network;
        self.segments = None # DataFrame; all segments on road network; several segments form a way (real road)
        self.ways = None # DataFrame; all ways (real road) on road network;
        
        self.adj_segments = None # DataFrame; relations of adjacent segments;
        self.adj_segments_graph = None # DiGraph; adjacent segment graph

        self.cellspace = None
        self.cell_distance_dic = {} # (cell_id_1, cell_id_2) -> mid_point_dis

        self.seg_startendid_to_incid = None
        self.seg_incid_to_startendid = None

        self.seg_dis_matrix = None # distance matrix
        self.seg_dis_matrix_segid_indices = None


    def __str__(self):
        return ''

    
    def load_data(self):
        if self.osm_raw_file == '':
            return -1
        _time = time.time()

        #======== basis for all =========
        self.nodes = pd.read_csv(self.node_file, delimiter = ',', index_col = 'node_id')

        # road segments dataset (to add additional features)
        # s_id, e_id, s_lon, s_lat, e_lon, e_lat, c_lon, c_lat,length, radian, highway_cls, way_ids, bridge, maxspeed,lanes, inc_id
        self.segments = pd.read_csv(self.segment_file, delimiter = ',', index_col = ['s_id','e_id'])
        self.segments.way_ids = self.segments.way_ids.apply(literal_eval)
        
        self.ways = pd.read_csv(self.way_file, delimiter = ',', index_col = 'way_id')
        self.ways.node_ids = self.ways.node_ids.apply(literal_eval)
        
        self.adj_segments = pd.read_csv(self.adj_segment_file, delimiter = ',')

        self.cellspace = pickle.load(open(self.cellspace_file, 'rb'))

        # (2653532349, 2653532346) -> {'inc_id': 15643}
        self.seg_startendid_to_incid = self.segments[['inc_id']].to_dict(orient = 'index')
        # 100 -> {'s_id': 276558558, 'e_id': 314623102}
        self.seg_incid_to_startendid = self.segments.reset_index().set_index('inc_id')[['s_id','e_id']].to_dict(orient = 'index')


        #======== more data processing =========

        # segment features
        _way_codes, _ = pd.factorize(self.segments.way_ids.map(lambda x: x[0]))
        self.segments['wayid_code'] = _way_codes

        _seg_codes, _ = pd.factorize(self.segments.inc_id)
        self.segments['segid_code'] = _seg_codes
        
        #####################
        ### FLOAT NUMBERS SMALLER THAN 1 CONVERTED INTO INTEGERS LARGER THAN 1
        ######################
        self.segments['length_code'] = (self.segments.length / Config.sarn_seg_length_unit).astype('int64')

        self.segments['radian_code'] = (self.segments.radian / Config.sarn_seg_radian_unit).astype('int64')

        self.segments['c_centrality_code'] = (self.segments.c_centrality * 10000000).astype('int64')    # NEW
        self.segments['b_centrality_code'] = (self.segments.b_centrality * 10000000).astype('int64')    # NEW
        self.segments['h_centrality_code'] = (self.segments.h_centrality * 10000000).astype('int64')    # NEW
        #self.segments['degree_code'] = (self.segments.degree * 1).astype('int64')    # NEW

        # add lat and lon code 
        _lon_unit = 50 * Config.dataset_lon2Euc_unit # 50 meters
        _lat_unit = 50 * Config.dataset_lat2Euc_unit
        cellspace_finegrained = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                                            self.cellspace.lon_max, self.cellspace.lat_max)
        self.segments['s_lon_code'] = ((self.segments.s_lon - cellspace_finegrained.lon_min) / cellspace_finegrained.lon_unit).astype('int64')
        self.segments['s_lat_code'] = ((self.segments.s_lat - cellspace_finegrained.lat_min) / cellspace_finegrained.lat_unit).astype('int64')
        self.segments['e_lon_code'] = ((self.segments.e_lon - cellspace_finegrained.lon_min) / cellspace_finegrained.lon_unit).astype('int64')
        self.segments['e_lat_code'] = ((self.segments.e_lat - cellspace_finegrained.lat_min) / cellspace_finegrained.lat_unit).astype('int64')

        # self.segments.lanes.loc[self.segments.lanes == 'NG'] = 0
        # replace NG values with 0
        self.segments.loc[(self.segments.lanes =='NG'), 'lanes'] = 0
        #self.segments.loc[(self.segments.c-centrality =='NG'), 'lanes'] = 0
        #self.segments.loc[(self.segments.b-centrality =='NG'), 'lanes'] = 0
        #self.segments.loc[(self.segments.h-centrality =='NG'), 'lanes'] = 0
        #self.segments.loc[(self.segments.degree =='NG'), 'lanes'] = 0
        self.segments.lanes = self.segments.lanes.astype('int32')
        self.segments.degree = self.segments.degree.astype('int32')    # NEW
        #self.segments.lanes

        # cellid of segment base on self.cellspace # caution, we have multiple cellspaces, ~= 1000meters
        if self.schema == 'SARN':
            _lon_unit = Config.dataset_lon2Euc_unit * Config.sarn_moco_multi_queue_cellsidelen
            _lat_unit = Config.dataset_lat2Euc_unit * Config.sarn_moco_multi_queue_cellsidelen
            self.cellspace = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                                            self.cellspace.lon_max, self.cellspace.lat_max)
            self.segments['c_cellid'] = self.segments[['c_lon','c_lat']].apply(lambda x: self.cellspace.get_cell_id_by_point(*x), axis=1)
            Config.sarn_moco_each_queue_size = math.ceil(Config.sarn_moco_total_queue_size / \
                                                    (self.cellspace.lon_size * self.cellspace.lat_size))

        # counting for Embedding constructor
        self.count_wayid_code = len(np.unique(self.segments['wayid_code'].values))
        self.count_segid_code = len(np.unique(self.segments['segid_code'].values))
        self.count_highway_cls = self.segments['highway_cls'].max() + 1
        self.count_length_code = self.segments['length_code'].max() + 1
        self.count_radian_code = self.segments['radian_code'].max() + 1
        self.count_s_lon_code = max(self.segments['s_lon_code'].max(), self.segments['e_lon_code'].max()) + 1
        self.count_s_lat_code = max(self.segments['s_lat_code'].max(), self.segments['e_lat_code'].max()) + 1
        self.count_lanes = self.segments['lanes'].max() + 1 # HRNR uses only
        #print(self.segments['lanes'].max())
        #print(self.segments['c-centrality'].max())
        self.count_c_centrality_code = self.segments['c_centrality_code'].max() + 1 # new
        self.count_b_centrality_code = self.segments['b_centrality_code'].max() + 1 # new
        self.count_h_centrality_code = self.segments['h_centrality_code'].max() + 1 # new
        self.count_degree_code = self.segments['degree'].max() + 1 # new

        # connected graph
        # 1. read adjacent segment graph from files
        # 2. add segment features to the adjacent graph
        # 3. verify connectness of graph, remove small subgraphs, create strongly/weakly? connected graph 
        self.adj_segments_graph = nx.from_pandas_edgelist(self.adj_segments, \
                                        's_id', 'e_id', edge_attr = True, create_using = nx.DiGraph())
        segment_attr_dicts = self.segments.reset_index().set_index('inc_id')[:].to_dict(orient = 'index')
        # https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html
        nx.set_node_attributes(self.adj_segments_graph, segment_attr_dicts)
        # connected graph
        # https://www.cnblogs.com/wushaogui/p/9204797.html
        adjsegs_subgraphs = sorted(nx.weakly_connected_components(self.adj_segments_graph), key = len, reverse = True)
        adjsegs_remove_nodes = set.union(*adjsegs_subgraphs[1: ])
        logging.debug('adj_segments_graph subgraphs={}, #remove_nodes={}'.format([len(x) for x in adjsegs_subgraphs], len(adjsegs_remove_nodes)))
        self.adj_segments_graph.remove_nodes_from(adjsegs_remove_nodes)
        logging.debug('adj_segments_graph. #nodes={}, #edge={}'.format( \
                        len(self.adj_segments_graph), self.adj_segments_graph.number_of_edges()))
        
        if self.schema == 'SARN':
            # 1. create distance-and-radian weighted segment graph. (do not add too many edge attributes)
            # 2. combine edge atrributes of spatial_segments_graph into self.adj_segments_graph
            spatial_segments_graph = self.__create_spatial_weighted_seg_graph()
            self.adj_segments_graph.add_edges_from(spatial_segments_graph.edges(data=True))

        # after all adj_segments_graph related operations. sorted
        self.segid_in_adj_segments_graph = sorted(list(self.adj_segments_graph.nodes()))
        self.seg_id_to_idx_in_adj_seg_graph = dict()
        self.seg_idx_to_id_in_adj_seg_graph = [-1] * len(self.segid_in_adj_segments_graph)
        for _idx, _id in enumerate(self.segid_in_adj_segments_graph):
            self.seg_id_to_idx_in_adj_seg_graph[_id] = _idx
            self.seg_idx_to_id_in_adj_seg_graph[_idx] = _id

        self.edge_index = EdgeIndex(self.adj_segments_graph, self.seg_id_to_idx_in_adj_seg_graph)

        # logging.info(self.segments[['inc_id','s_id','e_id']])    # debug
        # logging.info(len(self.segments[['inc_id','s_id','e_id']]))    # debug

        # segment features
        # self.seg_feats has dependency on self.segid_in_adj_segments_graph
        # , so dont move this part up
        ############
        # ADD NEW FEATURES HERE TOO
        ############
        _feat_columns = ['wayid_code', 'segid_code', 'highway_cls', 'length_code', \
                        'radian_code', 's_lon_code', 's_lat_code', 'e_lon_code', 'e_lat_code', 'lanes', 'c_centrality_code', 'b_centrality_code', 'h_centrality_code', 'degree']
        self.seg_feats = self.segments.reset_index().set_index('inc_id')[_feat_columns]
        self.seg_feats = torch.tensor(self.seg_feats.loc[self.segid_in_adj_segments_graph].values, dtype = torch.long, device = Config.device) # [N, n_feat_columns]

        logging.debug("seg_feats statistics: min={}, max={}".format( \
                        torch.min(self.seg_feats, dim = 0)[0].tolist(), \
                        torch.max(self.seg_feats, dim = 0)[0].tolist()))
   
        logging.info('osm_data load ended. @={:.3f}, #nodes={}, #segments={}, #ways={}, '
                        '#nodes_in_adj_seg={}, #edges_in_adj_seg={}'.format( \
                        time.time() - _time, self.nodes.shape[0], self.segments.shape[0], 
                        self.ways.shape[0], len(self.segid_in_adj_segments_graph), 
                        self.adj_segments_graph.number_of_edges()))


    def __create_spatial_weighted_seg_graph(self):
        # dont use self.cellspace here, because we may use a fine-grained cell space to represent lonlat.
        # For simplicity, we can create a new cellspace that the unit size 
        # is equal to Config.sarn_seg_weight_distance_thres, which will not slow down the program.

        _time = time.time()

        # ['inc_id_x', 'inc_id_y', 'distance', 'radian_delta', 'distance_weight', 'radian_weight', 'spatial_weight']
        all_seg_pairs = pd.DataFrame() 

        # create cellspace for 
        _lon_unit = Config.sarn_seg_weight_distance_thres * Config.dataset_lon2Euc_unit
        _lat_unit = Config.sarn_seg_weight_distance_thres * Config.dataset_lat2Euc_unit
        cs = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                        self.cellspace.lon_max, self.cellspace.lat_max)

        all_cell_pairs = cs.all_neighbour_cell_pairs_permutated() # all legal neighbouring cell pairs
        # a copy of road segments with attributes chosen
        # 4 SEGMENT FEATURES CHOSEN
        segs = self.segments[['inc_id','c_lon','c_lat','radian']].copy()
        segs.loc[:, 'c_cellid_tmp'] = segs.loc[:,['c_lon','c_lat']].apply(lambda x: cs.get_cell_id_by_point(*x), axis=1)

        for ((i_lon_1, i_lat_1), (i_lon_2, i_lat_2)) in all_cell_pairs:
            cell_id_1 = cs.get_cell_id(i_lon_1, i_lat_1)
            cell_id_2 = cs.get_cell_id(i_lon_2, i_lat_2)

            segs_in_cell_1 = segs[segs['c_cellid_tmp'] == cell_id_1]
            segs_in_cell_2 = segs[segs['c_cellid_tmp'] == cell_id_2]

            # 4 SEGMENT FEATURES CHOSEN
            segs_pairs = segs_in_cell_1[['inc_id','c_lon','c_lat','radian']].merge(segs_in_cell_2[['inc_id','c_lon','c_lat','radian']], how='cross')
            segs_pairs['distance'] = haversine_np(segs_pairs['c_lon_x'].values, segs_pairs['c_lat_x'].values, \
                                                    segs_pairs['c_lon_y'].values, segs_pairs['c_lat_y'].values)
            # remove all <seg_i, seg_i> self connections, distance thres, radian thres
            segs_pairs = segs_pairs[ (segs_pairs['distance'] <= Config.sarn_seg_weight_distance_thres) \
                                    & (segs_pairs['distance'] > 0) ]
            segs_pairs['radian_delta'] = np.abs(segs_pairs['radian_x'] - segs_pairs['radian_y'])
            segs_pairs = segs_pairs[ segs_pairs['radian_delta'] <= Config.sarn_seg_weight_radian_delta_thres ]
            segs_pairs = segs_pairs[['inc_id_x', 'inc_id_y', 'distance', 'radian_delta']]

            all_seg_pairs = pd.concat([all_seg_pairs, segs_pairs], axis = 0, ignore_index = True)

        # calculate distance weight and radian weight
        all_seg_pairs['distance_weight'] = np.cos(0.5 * np.pi * all_seg_pairs['distance'] / Config.sarn_seg_weight_distance_thres)
        all_seg_pairs['radian_weight'] = np.cos(0.5 * np.pi * all_seg_pairs['radian_delta'] / Config.sarn_seg_weight_radian_delta_thres)
        all_seg_pairs['spatial_weight'] = (all_seg_pairs['distance_weight'] + all_seg_pairs['radian_weight']) / 2
        
        # duplicate alone diagonal
        all_seg_pairs = pd.concat([all_seg_pairs, all_seg_pairs.rename(columns={'inc_id_x':'inc_id_y', 'inc_id_y':'inc_id_x'})], \
                                    axis = 0, ignore_index = True)
        
        # randomly sampling remove to allow the spatial_segments_graph
        # , having the same edge to self.adj_semgnts_graph 
        all_seg_pairs = all_seg_pairs.sample(n = self.adj_segments_graph.number_of_edges(),
                                            replace = False, weights = 'spatial_weight', 
                                            axis = 0)

        # convert to nx.graph
        spatial_segments_graph = nx.from_pandas_edgelist(all_seg_pairs, \
                                            'inc_id_x', 'inc_id_y', \
                                            edge_attr = True, create_using = nx.DiGraph())
        
        segs_remove_nodes = set(spatial_segments_graph.nodes).difference(set(self.adj_segments_graph.nodes))
        spatial_segments_graph.remove_nodes_from(segs_remove_nodes)
        segs_add_nodes = set(self.adj_segments_graph.nodes).difference(set(spatial_segments_graph.nodes))
        spatial_segments_graph.add_nodes_from(segs_add_nodes)
        logging.debug('spatial_adj_segments_graph #remove_nodes={}, #add_nodes={}'.format(len(segs_remove_nodes), len(segs_add_nodes)))

        # both graph should be have same number of nodes, even same nodes.
        # we have a loose assertion here. # TODO
        assert len(self.adj_segments_graph) == len(spatial_segments_graph) 
        
        logging.info('spatial_adj_segments_graph. #nodes={}, #edge={}, @={:.3f}'.format( \
                        len(spatial_segments_graph), spatial_segments_graph.number_of_edges(), \
                        time.time() - _time))

        return spatial_segments_graph
