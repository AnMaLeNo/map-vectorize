"""Vectorise un masque (A classique, ou B probabilités U-Net) et produit GeoJSON + aperçu du graphe."""
import cv2, json, sys, time, numpy as np
from mapvec import vectorize as V
from mapvec.methods import classical
src = sys.argv[1] if len(sys.argv) > 1 else 'A'
img = cv2.imread('data/nexus2011.jpg')
if src == 'A':
    mask = classical.segment(img)
else:
    mask = np.load(f'out/prob_{src}.npy').astype(np.float32) > float(sys.argv[2] if len(sys.argv) > 2 else 0.5)
t = time.time(); G, m, st = V.run(mask, src, f'out/graph_{src}.geojson'); print(src, st, f'{time.time() - t:.0f} s')
# aperçu : 2 tuiles d'évaluation, polygones en orange, arêtes en bleu, nœuds (rouge = carrefour, vert = cul-de-sac)
T = json.load(open('data/eval_tiles.json')); views = []
for k in ('vdg_dense', 'carrefour_morts'):
    x, y = T[k]; t_ = img[y:y+1024, x:x+1024].copy(); ov = t_.copy(); ov[m[y:y+1024, x:x+1024]] = (0, 160, 255)
    t_ = cv2.addWeighted(t_, 0.6, ov, 0.4, 0)
    for u, v, kk, d in G.edges(keys=True, data=True):
        pts = (d['xy'] - [x, y]).astype(np.int32)
        cv2.polylines(t_, [pts], False, (200, 60, 0), 3)
    for n, d in G.nodes(data=True):
        cx, cy = int(d['xy'][0] - x), int(d['xy'][1] - y)
        if 0 <= cx < 1024 and 0 <= cy < 1024:
            cv2.circle(t_, (cx, cy), 7, (0, 0, 220) if G.degree(n) >= 3 else (0, 170, 0), -1)
    views.append(t_)
cv2.imwrite(f'out/graph_{src}.jpg', cv2.resize(np.hstack(views), (2048, 1024)))
