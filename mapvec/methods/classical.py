"""Méthode A — segmentation classique : couleur (Lab) + morphologie.

Les galeries Nexus sont soit remplies de gris clair (grandes salles), soit simplement bordées de deux
murs gris moyen (couloirs étroits, intérieur blanc). On détecte donc :
  1. les murs : gris moyen peu saturé, plus foncé que le dessin de surface (gris pâle) et non bleu marine (texte) ;
  2. les couloirs : fermeture morphologique des murs, qui comble l'espace entre deux murs parallèles ;
  3. les salles : aplats gris clair d'une certaine épaisseur (ouverture, pour écarter les traits fins) ;
puis on réunit, on bouche les petits trous (textes posés sur une galerie) et on retire les petits fragments.
"""
import cv2
import numpy as np

DEFAULTS = dict(
    wall_L=(110, 195),     # luminance (0–255, OpenCV) des murs ; plus foncé = texte / symboles noirs
    max_chroma=7,          # murs : gris neutres
    text_b=-4,             # b* (centré) en dessous duquel un trait foncé est du texte bleu marine
    fill_L=(214, 251),     # luminance des aplats gris clair (très pâles : 244–247 ; papier : 253–255)
    fill_max_ab=2,         # aplats : gris strictement neutre (|a|,|b| ≤ 2), le bleu pâle des zones d'eau a b ≈ −4
    close_r=4,             # rayon de fermeture : comble les couloirs jusqu'à ~2·r px de large
    fill_open_r=3,         # épaisseur minimale d'un aplat pour compter comme salle
    hole_area=1500,        # trous bouchés (textes, symboles posés dans une galerie)
    min_area=600,          # fragments retirés
)


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(mask, max_area):
    inv = (~mask).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    return mask | small[lbl]


def remove_small(mask, min_area):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lbl]


def segment(img, **kw):
    p = {**DEFAULTS, **kw}
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.int16)
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128
    chroma = np.hypot(a, b)
    text = (L < 150) & (b < p['text_b'])
    wall = (L >= p['wall_L'][0]) & (L < p['wall_L'][1]) & (chroma < p['max_chroma']) & ~text
    corridors = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, disk(p['close_r'])) > 0
    fill = (L >= p['fill_L'][0]) & (L < p['fill_L'][1]) & (np.abs(a) <= p['fill_max_ab']) & (np.abs(b) <= p['fill_max_ab'])
    rooms = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_OPEN, disk(p['fill_open_r'])) > 0
    g = corridors | rooms
    g = fill_holes(g, p['hole_area'])
    return remove_small(g, p['min_area'])
