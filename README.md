# map-vectorize

Vectorisation automatique de cartes scannées — premier cas : Nexus « Alkhemia » 2011 (catacombes de Paris, 6307×6307 px).

## Évaluation
- 8 tuiles de 1024 px représentatives (`data/eval_tiles.json`), 480 points tirés en deux strates (près d'un trait / ailleurs), pondérés → estimations non biaisées sur les tuiles.
- Étiquetage visuel double vue (contexte + zoom ×6) ; 320 premiers points étiquetés deux fois (accord 94,7 %), désaccords tranchés au pixel ; 69 points « galerie ». `data/labels/labels_final.txt`.
- `mapvec/evaluate.py` : précision / rappel / IoU de la classe galerie, IC 95 % par bootstrap.
- Les 8 tuiles sont exclues de tout entraînement.

## Méthodes testées
| | Méthode | Précision | Rappel | IoU | Temps (M1) |
|---|---|---|---|---|---|
| A | Classique : couleur Lab + morphologie, halo de texte exclu (`methods/classical.py`) | 0,74 [0,61–0,87] | 0,50 [0,39–0,63] | 0,43 [0,32–0,55] | 1 s |
| SAM2 | SAM 2.1 small, zéro-shot, invites par points | — | — | — | non retenu : un clic donne soit un fragment, soit la moitié de la tuile |
| B | U-Net ResNet34 entraîné sur pseudo-étiquettes de A, seuil 0,6 (`methods/unet.py`) | 0,70 [0,57–0,82] | 0,58 [0,47–0,70] | 0,47 [0,36–0,58] | 1 h entraînement, 14 s inférence |

Note : le halo de texte de A (1 px) a été choisi sur le jeu d'évaluation ; légèrement optimiste.

## Vectorisation : graphe de points
`mapvec/vectorize.py` : masque → squelette → graphe (nœuds = carrefours / culs-de-sac, arêtes = axes de galerie avec longueur et largeur en m), élagage des barbules, polygones, géoréférencement WGS84 → GeoJSON. `python run_vec.py B 0.6`.

Les données (`data/`) et sorties (`out/`) ne sont pas versionnées, sauf les étiquettes.
