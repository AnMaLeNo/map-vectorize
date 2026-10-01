import cv2, numpy as np, torch, sys, time
from mapvec.methods import unet
from mapvec.evaluate import evaluate, fmt
img = cv2.imread('data/nexus2011.jpg')
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
y = unet.pseudo_labels(img); print('pseudo-étiquettes : galerie', (y == 1).mean().round(3), 'ignoré', (y == 255).mean().round(3))
model = unet.train(img, y, steps=steps)
torch.save(model.state_dict(), 'out/unet_b.pt')
t = time.time(); prob = unet.predict(model, img); print(f'inférence {time.time() - t:.0f} s')
np.save('out/prob_B.npy', prob.astype(np.float16))
for th in (0.3, 0.4, 0.5, 0.6):
    print(fmt(f'B U-Net seuil {th}', evaluate(prob > th)))
