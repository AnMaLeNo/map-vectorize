"""Masque de galeries → graphe de points (nœuds + arêtes) et polygones, géoréférencés en GeoJSON.

  1. nettoyage du masque (petits trous, fragments) ;
  2. squelette (axe médian) → graphe skan : nœuds = carrefours (degré ≥ 3) et culs-de-sac (degré 1) ;
  3. élagage itératif des barbules (branches terminales courtes, artefacts du squelette) ;
  4. arêtes simplifiées (Douglas-Peucker), avec longueur (m) et largeur moyenne (m, transformée de distance) ;
  5. polygones des galeries (contours avec trous) ;
  6. pixels → WGS84 par la transformation affine de géoréférencement de Nexus.
"""
import json

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Polygon, mapping
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

# Géoréférencement Nexus 2011 (README du dépôt catacombes, résidus ≈ ±8 m)
GEO = dict(ax=0.0000068431, bx=2.30265903, ay=-0.0000044980, by=48.84827524)
M_PER_PX = 0.0000068431 * 111320 * np.cos(np.radians(48.835))  # ≈ 0,50 m (x) ; y ≈ 0,0000044980·110540 ≈ 0,50 m


def px2geo(x, y):
    return round(GEO['ax'] * x + GEO['bx'], 7), round(GEO['ay'] * y + GEO['by'], 7)


def clean(mask, hole_area=300, min_area=800):
    m = mask.astype(np.uint8)
    inv = 1 - m
    n, lbl, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n, bool); small[1:] = st[1:, cv2.CC_STAT_AREA] <= hole_area
    m = (m.astype(bool) | small[lbl]).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lbl]


def skeleton_graph(mask, spur_px=12, simplify_px=1.5):
    """Graphe networkx : nœuds (x, y px), arêtes avec polyligne px, longueur px, largeur px."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    skel = skeletonize(mask)
    G = nx.MultiGraph()
    if not skel.any():
        return G
    S = Skeleton(skel)
    df = summarize(S, separator='_')
    for i, r in df.iterrows():
        pts = S.path_coordinates(i)  # (row, col)
        u, v = int(r['node_id_src']), int(r['node_id_dst'])
        for n in (u, v):
            if n not in G:
                G.add_node(n, xy=tuple(S.coordinates[n][::-1]))
        G.add_edge(u, v, rc=pts, length=float(r['branch_distance']),
                   width=float(2 * dist[pts[:, 0].astype(int), pts[:, 1].astype(int)].mean()))
    # élagage itératif des barbules : branche terminale plus courte que la demi-largeur locale + spur_px
    changed = True
    while changed:
        changed = False
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if u == v or not G.has_edge(u, v, k):
                continue
            if (G.degree(u) == 1) != (G.degree(v) == 1) and d['length'] < spur_px + d['width'] / 2:
                G.remove_edge(u, v, k)
                for n in (u, v):
                    if G.degree(n) == 0:
                        G.remove_node(n)
                changed = True
    # fusion des nœuds de degré 2 laissés par l'élagage (une galerie continue = une arête)
    for n in list(G.nodes):
        es = list(G.edges(n, keys=True, data=True)) if n in G else []
        if len(es) != 2 or G.number_of_edges(n, n):  # le degré change au fil des fusions : on le relit
            continue
        (_, a, _, da), (_, b, _, db) = es
        if a == b:  # boucle fermée sur un seul voisin : on la garde telle quelle
            continue
        pa = da['rc'] if np.allclose(da['rc'][-1], G.nodes[n]['xy'][::-1]) else da['rc'][::-1]
        pb = db['rc'] if np.allclose(db['rc'][0], G.nodes[n]['xy'][::-1]) else db['rc'][::-1]
        L = da['length'] + db['length']
        G.add_edge(a, b, rc=np.vstack([pa, pb]), length=L, width=(da['width'] * da['length'] + db['width'] * db['length']) / max(L, 1e-6))
        G.remove_node(n)
    for u, v, k, d in G.edges(keys=True, data=True):
        line = LineString(d['rc'][:, ::-1]).simplify(simplify_px)
        d['xy'] = np.asarray(line.coords)
    return G


def polygons(mask, simplify_px=1.0):
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1 or len(c) < 4:
            continue
        holes = []
        j = hier[0][i][2]
        while j != -1:
            if len(cs[j]) >= 4:
                holes.append(cs[j][:, 0, :])
            j = hier[0][j][0]
        p = Polygon(c[:, 0, :], holes).buffer(0).simplify(simplify_px)
        if not p.is_empty:
            out.append(p)
    return out


def to_geojson(G, polys, method):
    feats = []
    to_geo = lambda xy: [px2geo(x, y) for x, y in xy]
    for n, d in G.nodes(data=True):
        feats.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': px2geo(*d['xy'])},
                      'properties': {'kind': 'node', 'id': int(n), 'degree': G.degree(n),
                                     'role': 'carrefour' if G.degree(n) >= 3 else 'cul-de-sac' if G.degree(n) == 1 else 'autre'}})
    for u, v, k, d in G.edges(keys=True, data=True):
        feats.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': to_geo(d['xy'])},
                      'properties': {'kind': 'edge', 'u': int(u), 'v': int(v), 'length_m': round(d['length'] * M_PER_PX, 1),
                                     'width_m': round(d['width'] * M_PER_PX, 1)}})
    for p in polys:
        geo = mapping(p)
        def conv(rings):
            return [to_geo(r) for r in rings]
        if geo['type'] == 'Polygon':
            geo = {'type': 'Polygon', 'coordinates': conv(geo['coordinates'])}
        else:
            geo = {'type': 'MultiPolygon', 'coordinates': [conv(pp) for pp in geo['coordinates']]}
        feats.append({'type': 'Feature', 'geometry': geo, 'properties': {'kind': 'gallery'}})
    return {'type': 'FeatureCollection', 'properties': {'source': 'Nexus Alkhemia 2011', 'method': method,
                                                       'status': 'automatique, non vérifié'}, 'features': feats}


def stats(G):
    deg = dict(G.degree())
    return {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(),
            'junctions': sum(d >= 3 for d in deg.values()), 'dead_ends': sum(d == 1 for d in deg.values()),
            'components': nx.number_connected_components(G),
            'length_km': round(sum(d['length'] for *_, d in G.edges(data=True)) * M_PER_PX / 1000, 2)}


def run(mask, method, out_path):
    m = clean(mask)
    G = skeleton_graph(m)
    gj = to_geojson(G, polygons(m), method)
    json.dump(gj, open(out_path, 'w'))
    return G, m, stats(G)
