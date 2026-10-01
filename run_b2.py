import cv2, numpy as np, torch, sys, time
from mapvec.methods import unet
from mapvec.evaluate import evaluate, fmt
img = cv2.imread('data/nexus2011.jpg'); prob1 = np.load('out/prob_B.npy').astype(np.float32)
y = unet.pseudo_labels_round2(img, prob1); print('étiquettes tour 2 : galerie', (y == 1).mean().round(3), 'fond', (y == 0).mean().round(3), 'ignoré', (y == 255).mean().round(3), flush=True)
model = unet.train(img, y, steps=int(sys.argv[1]), seed=1)
torch.save(model.state_dict(), 'out/unet_b2.pt')
prob = unet.predict(model, img); np.save('out/prob_B2.npy', prob.astype(np.float16))
for th in (0.4, 0.5, 0.6, 0.7): print(fmt(f'B2 seuil {th}', evaluate(prob > th)), flush=True)
