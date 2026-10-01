import cv2, json, time, numpy as np
from mapvec.methods import classical
from mapvec.evaluate import evaluate, fmt
img = cv2.imread('data/nexus2011.jpg')
t = time.time(); m = classical.segment(img); dt = time.time() - t
r = evaluate(m); print(fmt('A classique', r), f'  ({dt:.0f} s)'); print('erreurs', r['errors'])
np.save('out/mask_A.npy', m)
T = json.load(open('data/eval_tiles.json')); ths = []
for k, (x, y) in T.items():
    t = img[y:y+1024, x:x+1024].copy(); ov = t.copy(); ov[m[y:y+1024, x:x+1024]] = (0, 140, 255)
    t = cv2.addWeighted(t, 0.55, ov, 0.45, 0); cv2.putText(t, k, (8, 30), 0, 1, (0, 0, 200), 2); ths.append(cv2.resize(t, (512, 512), interpolation=cv2.INTER_AREA))
cv2.imwrite('out/A_overlay.jpg', np.vstack([np.hstack(ths[:4]), np.hstack(ths[4:])]))
