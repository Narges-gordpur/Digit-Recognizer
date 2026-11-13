# train.py
import os, glob, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torch.nn as nn

# ---------- تنظیمات ----------
IMG_H, IMG_W = 32, 256
MAX_DIGITS   = 8
BLANK        = 0
NUM_CLASSES  = 11          # blank + 10 رقم

# ---------- فونت‌ها ----------
def find_fonts():
    pats = [
        '/usr/share/fonts/**/*.ttf', '/usr/share/fonts/**/*.otf',
        '/usr/local/share/fonts/**/*.ttf',
        'C:/Windows/Fonts/*.ttf',
        os.path.expanduser('~/Library/Fonts/**/*.ttf'),
        '/System/Library/Fonts/**/*.ttf',
        '/System/Library/Fonts/Supplemental/**/*.ttf',
    ]
    fs = []
    for p in pats:
        fs += glob.glob(p, recursive=True)
    return fs

FONTS = find_fonts()
print(f"[i] {len(FONTS)} فونت پیدا شد.")

def _get_font(size):
    if FONTS:
        try:
            return ImageFont.truetype(random.choice(FONTS), size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

# ---------- تولید نمونه مصنوعی ----------
def make_sample(text=None):
    if text is None:
        n = random.randint(1, MAX_DIGITS)
        text = ''.join(random.choices('0123456789', k=n))

    fsize = random.randint(18, 30)
    font  = _get_font(fsize)

    bw, bh = IMG_W * 2, IMG_H * 2
    canvas = Image.new('L', (bw, bh), 255)
    draw   = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw > IMG_W - 12:
        fsize = max(10, int(fsize * (IMG_W - 12) / tw))
        font  = _get_font(fsize)
        bbox  = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (bw - tw) // 2 - bbox[0]
    y = (bh - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=random.randint(0, 70), font=font)

    ang = random.uniform(-2, 2)
    if abs(ang) > 0.3:
        canvas = canvas.rotate(ang, resample=Image.BILINEAR, fillcolor=255)

    left = (bw - IMG_W) // 2 + random.randint(-12, 12)
    top  = (bh - IMG_H) // 2 + random.randint(-2, 2)
    left = max(0, min(bw - IMG_W, left))
    top  = max(0, min(bh - IMG_H, top))
    img  = canvas.crop((left, top, left + IMG_W, top + IMG_H))

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.7)))
    arr = np.array(img, dtype=np.float32)
    if random.random() < 0.5:
        arr = np.clip(arr + np.random.normal(0, random.uniform(1, 5), arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), text

# ---------- کدگذاری ----------
def encode(text):
    return torch.tensor([int(c) + 1 for c in text], dtype=torch.long)

def to_tensor(img):
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)

def ctc_greedy_decode(log_probs):
    idx = log_probs.argmax(dim=2).cpu().numpy()
    out_all = []
    for seq in idx:
        res, prev = [], -1
        for i in seq:
            i = int(i)
            if i != BLANK and i != prev:
                res.append(str(i - 1))
            prev = i
        out_all.append(''.join(res))
    return out_all

# ---------- مدل CRNN ----------
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2)
        f = f.permute(0, 2, 1)
        f, _ = self.lstm(f)
        f = self.drop(f)
        return self.fc(f).log_softmax(2)

# ---------- آموزش ----------
def collate(batch):
    imgs  = torch.stack([b[0] for b in batch], 0)
    tgts  = [b[1] for b in batch]
    lens  = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    return imgs, torch.cat(tgts), lens

def make_batch(bs):
    items = [make_sample() for _ in range(bs)]
    batch = [(to_tensor(img), encode(txt)) for img, txt in items]
    return collate(batch)

@torch.no_grad()
def evaluate(model, device, n=150):
    model.eval()
    correct = 0
    for _ in range(n):
        k = random.randint(1, MAX_DIGITS)
        text = ''.join(random.choices('0123456789', k=k))
        img, _ = make_sample(text)
        pred = ctc_greedy_decode(model(to_tensor(img).unsqueeze(0).to(device)))[0]
        correct += (pred == text)
    model.train()
    return correct / n

def train(epochs=25, steps=150, bs=64, lr=3e-4, ckpt='crnn_digits.pt'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[i] device = {device}")

    model = CRNN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    ctc   = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step in range(steps):
            imgs, tgts, tgt_lens = make_batch(bs)
            imgs, tgts, tgt_lens = imgs.to(device), tgts.to(device), tgt_lens.to(device)

            logits = model(imgs)
            T      = logits.size(1)
            in_lens = torch.full((bs,), T, dtype=torch.long, device=device)
            loss = ctc(logits.permute(1, 0, 2), tgts, in_lens, tgt_lens)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            running += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  ep {ep} step {step+1}/{steps}  loss={running/50:.3f}")
                running = 0.0

        acc = evaluate(model, device)
        print(f"[OK] epoch {ep}: exact-match accuracy = {acc*100:.1f}%")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)
            print(f"    -> ذخیره شد ({ckpt})")

    print(f"[OK] پایان. بهترین دقت: {best*100:.1f}%")

if __name__ == '__main__':
    train()