"""Build the SARN-schema dataset for Rome from the project's simplified network.

The project's emissions live on the osmnx-simplified undirected network of
95,848 edges (see rome_highway_audit.py -> highway_audit_edges.parquet).  This
converter reproduces that exact network in SARN's schema -- one DIRECTED segment
per direction per edge -- so SARN embeddings map 1:1 back onto the emission
edges.  Endpoint coordinates come from the centro PBF; length / road class from
the audit table.

Outputs (prefix = OSM_Rome_raw, written to data/):
    <prefix>_node          node_id,lon,lat            (segment endpoints)
    <prefix>_segment       ... (2 directed segments per edge, 191,696 total)
    <prefix>_way           synthetic ways (one per edge)
    <prefix>_adjsegment    dual-graph adjacency
    <prefix>_cellspace_info.pickle

Run with .venv-osm (pyosmium + pyarrow).
"""
from __future__ import annotations

import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import osmium
import pandas as pd

SARN_DIR = Path("/Users/gurban.aliyevuni-a.de/workspace/unipi/gps2exposure/SARN")
sys.path.insert(0, str(SARN_DIR))
from utils.cell import CellSpace  # noqa: E402

PBF = Path("/Users/gurban.aliyevuni-a.de/workspace/unipi/PhD_Thesis_Gurban_Aliyev/data/rome/network/osm/centro-latest.osm.pbf")
EDGES = Path("/Users/gurban.aliyevuni-a.de/workspace/unipi/PhD_Thesis_Gurban_Aliyev/data/rome/imputed/v3/audit/highway_audit_edges.parquet")
MAP_FEATURES = Path("/Users/gurban.aliyevuni-a.de/workspace/unipi/PhD_Thesis_Gurban_Aliyev/data/rome/imputed/v3/map_complete/map_features.parquet")
BH_CENTR = Path("/Users/gurban.aliyevuni-a.de/workspace/unipi/gps2exposure/data_imputation/data/rome_simple_undirected_network_edge_b_h_centralities.csv")
OUT_PREFIX = SARN_DIR / "data" / "OSM_Rome_raw"

POI_CATS = ["cultural", "education", "food", "health", "service", "transportation"]

LAT_LO, LON_LO, LAT_HI, LON_HI = 41.60, 12.15, 42.15, 12.80

# highway string -> SARN class (0..5)
HIGHWAY_CLS = {
    "motorway": 0, "motorway_link": 0, "trunk": 0, "trunk_link": 0,
    "primary": 1, "primary_link": 1,
    "secondary": 2, "secondary_link": 2,
    "tertiary": 3, "tertiary_link": 3, "unclassified": 3,
    "residential": 4, "living_street": 4,
    "service": 5, "road": 5,
}
HIGHWAY_WEIGHT = {"0": 6.0, "1": 5.0, "2": 4.0, "3": 3.0, "4": 2.0, "5": 1.0}


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return round(2 * math.asin(math.sqrt(a)) * 6367 * 1000, 3)


def bearing(lon1, lat1, lon2, lat2):
    dy = lat2 - lat1
    dx = lon2 - lon1
    if dx == 0:
        return round(math.pi / 2, 3) if dy >= 0 else round(1.5 * math.pi, 3)
    r = math.atan(dy / dx)
    if dx < 0:
        r += math.pi
    elif dy < 0:
        r += 2 * math.pi
    return round(r, 3)


class NodeReader(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.coords = {}

    def node(self, n):
        loc = n.location
        if loc.valid() and LON_LO <= loc.lon <= LON_HI and LAT_LO <= loc.lat <= LAT_HI:
            self.coords[n.id] = (loc.lon, loc.lat)


def main():
    print("parsing PBF nodes ...", flush=True)
    nr = NodeReader()
    nr.apply_file(str(PBF))
    coords = nr.coords
    print(f"  nodes={len(coords)}", flush=True)

    g = pd.read_parquet(EDGES)
    print(f"  edges={len(g)}", flush=True)

    # class-median length fallback for edges with no geometry
    resolved = g[g.length_osm_m.notna() & (g.length_osm_m > 0)]
    class_med_len = resolved.groupby("hw_group").length_osm_m.median()
    fallback_len = class_med_len.median()

    def length_of(r):
        if pd.notna(r.length_m) and r.length_m > 0:
            return r.length_m
        if pd.notna(r.length_osm_m) and r.length_osm_m > 0:
            return r.length_osm_m
        return class_med_len.get(r.hw_group, fallback_len)

    g["length_used"] = g.apply(length_of, axis=1)
    g["cls"] = g["highway"].map(HIGHWAY_CLS).fillna(3).astype(int)

    # ---- extended segment features (fork's osm_loader expects these) ------
    mf = pd.read_parquet(MAP_FEATURES, columns=["a", "b", "c_centrality"]
                         + [f"{c}_500" for c in POI_CATS])
    g = g.merge(mf, on=["a", "b"], how="left")
    for c in POI_CATS:
        g[c] = g[f"{c}_500"]
    # node-level b/h centrality -> segment (mean of endpoints)
    bh = pd.read_csv(BH_CENTR)
    bh = bh.rename(columns={bh.columns[0]: "node"})
    bh["node"] = pd.to_numeric(bh["node"], errors="coerce")
    bh = bh.dropna(subset=["node"]).astype({"node": "int64"}).set_index("node")
    bmap = bh["b_centrality"].to_dict()
    hmap = bh["h_centrality"].to_dict()
    g["b_centrality"] = (g.a.map(bmap).fillna(0.0) + g.b.map(bmap).fillna(0.0)) / 2
    g["h_centrality"] = (g.a.map(hmap).fillna(0.0) + g.b.map(hmap).fillna(0.0)) / 2
    # degree = sum of endpoint degrees
    deg = pd.concat([g.a, g.b]).value_counts()
    g["degree"] = (g.a.map(deg).fillna(0) + g.b.map(deg).fillna(0)).astype(int)
    ext_cols = ["c_centrality", "b_centrality", "h_centrality", "degree"] + POI_CATS
    g[ext_cols] = g[ext_cols].fillna(0)  # unmatched -> unclassified

    # ---- emit directed segments -------------------------------------------
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    inc_id = 100
    seg_inc = {}
    seg_meta = {}
    used_nodes = set()

    with open(str(OUT_PREFIX) + "_segment", "w") as f:
        f.write("s_id,e_id,s_lon,s_lat,e_lon,e_lat,c_lon,c_lat,length,radian,"
                "highway_cls,way_ids,bridge,maxspeed,lanes,inc_id,"
                "c_centrality,b_centrality,h_centrality,degree,"
                "cultural,education,food,health,service,transportation\n")
        for row in g.itertuples():
            a, b = int(row.a), int(row.b)
            if a == b or a not in coords or b not in coords:
                continue  # skip self-loops and edges with missing geometry
            used_nodes.add(a); used_nodes.add(b)
            (alon, alat), (blon, blat) = coords[a], coords[b]
            brg_ab = bearing(alon, alat, blon, blat)
            brg_ba = round((brg_ab + math.pi) % (2 * math.pi), 3)
            ext = (f"{row.c_centrality},{row.b_centrality},{row.h_centrality},"
                   f"{int(row.degree)},{int(row.cultural)},{int(row.education)},"
                   f"{int(row.food)},{int(row.health)},{int(row.service)},"
                   f"{int(row.transportation)}")
            for (s_id, e_id, slon, slat, elon, elat, brg) in (
                (a, b, alon, alat, blon, blat, brg_ab),
                (b, a, blon, blat, alon, alat, brg_ba),
            ):
                clon, clat = round((slon + elon) / 2, 8), round((slat + elat) / 2, 8)
                seg_inc[(s_id, e_id)] = inc_id
                seg_meta[(s_id, e_id)] = {"inc_id": inc_id, "cls": int(row.cls), "length": row.length_used}
                f.write(f"{s_id},{e_id},{slon},{slat},{elon},{elat},{clon},{clat},"
                        f"{row.length_used},{brg},{int(row.cls)},"
                        f"\"[{inc_id}]\",NG,NG,0,{inc_id},{ext}\n")
                inc_id += 1

    n_seg = len(seg_inc)
    print(f"  directed segments={n_seg}", flush=True)

    # ---- node file --------------------------------------------------------
    with open(str(OUT_PREFIX) + "_node", "w") as f:
        f.write("node_id,lon,lat\n")
        for nid in sorted(used_nodes):
            lon, lat = coords[nid]
            f.write(f"{nid},{lon},{lat}\n")

    # ---- way file (one synthetic way per directed segment) ----------------
    with open(str(OUT_PREFIX) + "_way", "w") as f:
        f.write("way_id,node_ids,highway,highway_cls,one_way,bridge,maxspeed,lanes\n")
        for (s_id, e_id), meta in seg_meta.items():
            f.write(f"{meta['inc_id']},\"[{s_id},{e_id}]\",road,{meta['cls']},1,NG,NG,NG\n")

    # ---- adjsegment (dual graph) ------------------------------------------
    node_in = defaultdict(set)
    node_out = defaultdict(set)
    for (s_id, e_id) in seg_inc:
        node_out[s_id].add(e_id)
        node_in[e_id].add(s_id)
    with open(str(OUT_PREFIX) + "_adjsegment", "w") as f:
        f.write("s1s_id,s1e_id,s2s_id,s2e_id,s_id,e_id,weight,s1_length\n")
        for mid in used_nodes:
            left_set = node_in.get(mid)
            right_set = node_out.get(mid)
            if not left_set or not right_set:
                continue
            for left in left_set:
                for right in right_set:
                    if left == right and len(left_set) == 2 and len(right_set) == 2:
                        continue
                    w1 = HIGHWAY_WEIGHT[str(seg_meta[(left, mid)]["cls"])]
                    w2 = HIGHWAY_WEIGHT[str(seg_meta[(mid, right)]["cls"])]
                    weight = w1 + w2
                    if left == right:
                        weight /= 10
                    s1_length = seg_meta[(left, mid)]["length"]
                    f.write(f"{left},{mid},{mid},{right},"
                            f"{seg_inc[(left, mid)]},{seg_inc[(mid, right)]},"
                            f"{weight},{s1_length}\n")

    # ---- cellspace --------------------------------------------------------
    lons = [coords[n][0] for n in used_nodes]
    lats = [coords[n][1] for n in used_nodes]
    lon_min, lon_max, lat_min, lat_max = min(lons), max(lons), min(lats), max(lats)
    lon_unit = 1.0 / (111320 * math.cos(math.radians(41.9))) * 1000
    lat_unit = 1.0 / 110540 * 1000
    cellspace = CellSpace(lon_unit, lat_unit, lon_min, lat_min, lon_max, lat_max)
    with open(str(OUT_PREFIX) + "_cellspace_info.pickle", "wb") as f:
        pickle.dump(cellspace, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"done: edges={len(used_nodes)} nodes, {n_seg} directed segments, "
          f"cellspace={cellspace.lon_size}x{cellspace.lat_size}")


if __name__ == "__main__":
    main()
