"""Render turtlesim-style demo videos from the packages' own logic (offline)."""
import math, os, random, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'videos')
FPS = 30
DT = 1.0 / FPS
W = 600
PANEL = 150
SCALE = W / 11.088889
BG = (69, 86, 255)
FONT = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf', 17)
FONT_S = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf', 13)
FONT_B = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans-Bold.ttf', 20)
FONT_T = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans-Bold.ttf', 30)


def px(x, y):
    return x * SCALE, W - y * SCALE


def clamp(v, a, b):
    return max(a, min(b, v))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class T:
    """Unicycle turtle matching turtlesim kinematics."""
    def __init__(self, name, x, y, th, shell, pen):
        self.name, self.x, self.y, self.th, self.v = name, x, y, th, 0.0
        self.shell, self.pen, self.trail = shell, pen, [(x, y)]

    def step(self, lin, ang, dt=DT):
        self.th = wrap(self.th + ang * dt)
        self.x = clamp(self.x + lin * math.cos(self.th) * dt, 0, 11.088889)
        self.y = clamp(self.y + lin * math.sin(self.th) * dt, 0, 11.088889)
        self.v = lin
        if self.pen:
            self.trail.append((self.x, self.y))


def draw_turtle(d, t):
    cx, cy = px(t.x, t.y)
    c, s = math.cos(t.th), -math.sin(t.th)
    def rot(u, v):
        return cx + u * c - v * s, cy + u * s + v * c
    for (u, v) in ((9, 9), (9, -9), (-9, 9), (-9, -9)):  # flippers
        x0, y0 = rot(u, v)
        d.ellipse([x0 - 4, y0 - 4, x0 + 4, y0 + 4], fill=(60, 140, 60))
    hx, hy = rot(15, 0)
    d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=(90, 170, 90))
    shell = [rot(13 * math.cos(a), 10 * math.sin(a))
             for a in [i * math.pi / 10 for i in range(20)]]
    d.polygon(shell, fill=t.shell, outline=(30, 30, 30))
    d.text((cx, cy - 26), t.name, font=FONT_S, fill='white', anchor='mm')


class Video:
    def __init__(self, path):
        os.makedirs(OUT, exist_ok=True)
        self.p = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
             '-pix_fmt', 'rgb24', '-s', f'{W}x{W + PANEL}', '-r', str(FPS),
             '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf',
             '23', path], stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, img, repeat=1):
        b = img.tobytes()
        for _ in range(repeat):
            self.p.stdin.write(b)
            self.frames += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        return self.frames / FPS


def wrap_text(text, font, width, d):
    words, lines, cur = text.split(), [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if d.textlength(trial, font=font) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def frame(turtles, obstacles, header, status, caption, extra=None,
          banner=None):
    img = Image.new('RGB', (W, W + PANEL), BG)
    d = ImageDraw.Draw(img)
    for ox, oy, r in obstacles:
        x0, y0 = px(ox - r, oy + r)
        x1, y1 = px(ox + r, oy - r)
        d.ellipse([x0, y0, x1, y1], fill=(40, 40, 40))
    for t in turtles:
        if t.pen and len(t.trail) > 1:
            d.line([px(*p) for p in t.trail], fill=t.pen, width=2)
    if extra:
        extra(d)
    for t in turtles:
        draw_turtle(d, t)
    d.rectangle([0, W, W, W + PANEL], fill=(25, 25, 35))
    d.text((12, W + 8), header, font=FONT_B, fill=(255, 220, 120))
    d.text((12, W + 36), status, font=FONT, fill='white')
    y = W + 64
    for line in wrap_text(caption, FONT_S, W - 24, d)[:5]:
        d.text((12, y), line, font=FONT_S, fill=(200, 200, 210))
        y += 17
    if banner:
        bw = d.textlength(banner, font=FONT_T) + 40
        d.rectangle([(W - bw) / 2, W / 2 - 30, (W + bw) / 2, W / 2 + 30],
                    fill=(0, 0, 0))
        d.text((W / 2, W / 2), banner, font=FONT_T, fill='white',
               anchor='mm')
    return img


def title_card(video, lines, secs=3.0):
    img = Image.new('RGB', (W, W + PANEL), (20, 22, 40))
    d = ImageDraw.Draw(img)
    y = 180
    for i, line in enumerate(lines):
        f = FONT_T if i == 0 else FONT
        for sub in wrap_text(line, f, W - 60, d):
            d.text((W / 2, y), sub, font=f, fill='white', anchor='mm')
            y += 44 if i == 0 else 26
        y += 8
    d.text((W / 2, W + PANEL - 30),
           'Offline render of the ROS 2 package logic (not a turtlesim '
           'screen capture)', font=FONT_S, fill=(150, 150, 170), anchor='mm')
    video.write(img, int(secs * FPS))


def caption_at(captions, t):
    text = ''
    for start, c in captions:
        if t >= start:
            text = c
    return text
