"""Évaluation par points étiquetés (data/labels) : précision, rappel et IoU de la classe « galerie ».

Les points sont tirés en deux strates (A : près d'un trait, B : ailleurs) ; chaque point porte un poids
= nombre de pixels de sa strate qu'il représente, ce qui donne des estimations non biaisées sur les tuiles.
Intervalles de confiance à 95 % par bootstrap.

Étiquettes : G (galerie étage sup.), L (étage inf.) → positif ; N, B (zone bleue), Z (ossuaire) → négatif ;
X (encart), C (chevrons), ? → exclus.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS, NEG = {'G', 'L'}, {'N', 'B', 'Z'}


def load_points():
    pts = json.load(open(os.path.join(ROOT, 'data/labels/points.json')))
    lab = {}
    for line in open(os.path.join(ROOT, 'data/labels/labels_final.txt')):
        if not line.startswith('#'):
            i, v = line.split()
            lab[int(i)] = v
    for p in pts:
        p['label'] = lab[p['id']]
    return [p for p in pts if p['label'] in POS | NEG]


def _scores(y, yhat, w):
    tp = (w * (y & yhat)).sum()
    fp = (w * (~y & yhat)).sum()
    fn = (w * (y & ~yhat)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return prec, rec, iou


def evaluate(mask, n_boot=2000, seed=0):
    """mask : tableau booléen pleine résolution (True = galerie). Renvoie un dict de métriques + IC 95 %."""
    pts = load_points()
    y = np.array([p['label'] in POS for p in pts])
    yhat = np.array([bool(mask[p['y'], p['x']]) for p in pts])
    w = np.array([p['w'] for p in pts])
    est = _scores(y, yhat, w)
    rng = np.random.default_rng(seed)
    boot = np.array([_scores(y[i], yhat[i], w[i]) for i in (rng.integers(0, len(pts), len(pts)) for _ in range(n_boot))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    out = {k: (round(float(e), 3), round(float(a), 3), round(float(b), 3)) for k, e, a, b in zip(('precision', 'recall', 'iou'), est, lo, hi)}
    out['n_points'] = len(pts)
    out['n_positive'] = int(y.sum())
    out['errors'] = {'false_pos': [p['id'] for p, a, b in zip(pts, y, yhat) if b and not a],
                     'false_neg': [p['id'] for p, a, b in zip(pts, y, yhat) if a and not b]}
    return out


def fmt(name, r):
    f = lambda k: f"{r[k][0]:.2f} [{r[k][1]:.2f}–{r[k][2]:.2f}]"
    return f"{name:<28} précision {f('precision')}  rappel {f('recall')}  IoU {f('iou')}"
