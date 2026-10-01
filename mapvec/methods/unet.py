"""Méthode B — U-Net (encodeur ResNet34 ImageNet) entraîné sur des pseudo-étiquettes.

Étiquettes d'entraînement tirées de la méthode A, en trois états : galerie (1), fond (0), incertain (255,
ignoré par la perte) — bande de 3 px autour des contours de A et voisinage du texte. Les 8 tuiles
d'évaluation (+ marge) sont exclues de l'entraînement.
"""
import json, os, time
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from . import classical

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def eval_exclusion(shape, margin=64):
    ex = np.zeros(shape[:2], bool)
    for x, y in json.load(open(os.path.join(ROOT, 'data/eval_tiles.json'))).values():
        ex[max(0, y - margin):y + 1024 + margin, max(0, x - margin):x + 1024 + margin] = True
    return ex


def pseudo_labels(img, band=3):
    a = classical.segment(img).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    text = (lab[..., 0] < 150) & (lab[..., 2] - 128 < -4)
    y = a.copy()
    k = classical.disk(band)
    edge = (cv2.dilate(a, k) > 0) & ~(cv2.erode(a, k) > 0)
    y[edge] = 255
    y[cv2.dilate(text.astype(np.uint8), classical.disk(3)) > 0] = 255
    return y


def to_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


def make_model():
    return smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)


def train(img, y, steps=3000, crop=384, bs=8, lr=3e-4, seed=0, log=print):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ex = eval_exclusion(img.shape)
    H, W = y.shape
    model = make_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    # positions d'échantillonnage : la moitié centrées près des galeries (classe minoritaire)
    pos_yx = np.argwhere((y == 1) & ~ex)
    t0 = time.time()
    for step in range(steps):
        xb, yb = [], []
        while len(xb) < bs:
            if rng.random() < 0.5:
                cy, cx = pos_yx[rng.integers(len(pos_yx))]; y0 = int(np.clip(cy - crop // 2, 0, H - crop)); x0 = int(np.clip(cx - crop // 2, 0, W - crop))
            else:
                y0 = int(rng.integers(0, H - crop)); x0 = int(rng.integers(0, W - crop))
            if ex[y0:y0 + crop, x0:x0 + crop].any():
                continue
            im = rgb[y0:y0 + crop, x0:x0 + crop]; lb = y[y0:y0 + crop, x0:x0 + crop]
            k = int(rng.integers(4)); im = np.rot90(im, k); lb = np.rot90(lb, k)
            if rng.random() < 0.5: im = im[:, ::-1]; lb = lb[:, ::-1]
            im = np.clip(im.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
            xb.append(to_tensor(np.ascontiguousarray(im))); yb.append(torch.from_numpy(np.ascontiguousarray(lb)))
        xb = torch.stack(xb).to(DEV); yb = torch.stack(yb).to(DEV)
        logit = model(xb)[:, 0]
        valid = yb != 255
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[valid], yb[valid].float())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            log(f'  pas {step:5d}  perte {loss.item():.4f}  {time.time() - t0:.0f} s')
    return model


@torch.no_grad()
def predict(model, img, tile=1024, overlap=128):
    """Probabilité de galerie sur toute l'image (tuiles chevauchantes, fenêtre de pondération)."""
    model.eval()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    w1 = np.minimum(np.arange(tile) + 1, tile - np.arange(tile)).astype(np.float32); w1 = np.minimum(w1, overlap) / overlap
    win = np.outer(w1, w1)
    step = tile - overlap
    for y0 in list(range(0, H - tile, step)) + [H - tile]:
        for x0 in list(range(0, W - tile, step)) + [W - tile]:
            p = torch.sigmoid(model(to_tensor(rgb[y0:y0 + tile, x0:x0 + tile])[None].to(DEV))[0, 0]).cpu().numpy()
            acc[y0:y0 + tile, x0:x0 + tile] += p * win; wsum[y0:y0 + tile, x0:x0 + tile] += win
    return acc / np.maximum(wsum, 1e-6)


def pseudo_labels_round2(img, prob, band=2, hi=0.8, lo=0.2):
    """Tour 2 (auto-apprentissage) : galerie là où A et le U-Net du tour 1 sont d'accord avec confiance ;
    fond là où les deux disent non — y compris le texte hors galerie, explicitement négatif cette fois."""
    a = classical.segment(img)
    y = np.full(a.shape, 255, np.uint8)
    y[(prob > hi) & a] = 1
    y[(prob < lo) & ~a] = 0
    k = classical.disk(band)
    pos = (y == 1).astype(np.uint8)
    edge = (cv2.dilate(pos, k) > 0) & ~(cv2.erode(pos, k) > 0)
    y[edge] = 255
    return y
