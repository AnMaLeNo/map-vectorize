# map-vectorize

Vectorisation automatique de cartes scannées — premier cas : Nexus « Alkhemia » 2011 (catacombes de Paris, 6307×6307 px).

## Évaluation
- 8 tuiles de 1024 px représentatives (`data/eval_tiles.json`), 320 points tirés en deux strates (près d'un trait / ailleurs), pondérés.
- Étiquetage visuel en deux passes indépendantes (accord 94,7 %), désaccords tranchés au pixel : `data/labels/labels_final.txt`.
- `mapvec/evaluate.py` : précision / rappel / IoU de la classe galerie, IC 95 % par bootstrap.
- Limite : 29 points positifs seulement → intervalles larges ; à densifier avant de départager des méthodes proches.

## Méthodes
| | Méthode | Précision | Rappel | IoU | Temps |
|---|---|---|---|---|---|
| A | Classique : couleur Lab + morphologie (`mapvec/methods/classical.py`) | 0,50 [0,33–0,69] | 0,55 [0,36–0,73] | 0,35 [0,21–0,51] | 1 s (M1) |

Les données (`data/`) et sorties (`out/`) ne sont pas versionnées.
