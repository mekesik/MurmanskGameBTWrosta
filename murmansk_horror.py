#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Роста блять — ПОЛЯРНАЯ НОЧЬ
Полноценная 3D-хоррор-игра на Ursina с кооперативом на 2 игрока.

Запуск:
    python murmansk_horror.py                 # меню (хост / клиент / одиночная)
    python murmansk_horror.py --host          # сразу сервер
    python murmansk_horror.py --client 26.x.x.x
    python murmansk_horror.py --single
    python murmansk_horror.py --port 7778
    python murmansk_horror.py --smoke-test    # автотест (одиночный)

Сетевые порты: TCP 7778 (можно изменить через --port).
Для Radmin VPN хост запускает сервер и сообщает свой Radmin-IP клиенту.
"""

from __future__ import annotations

# Ursina импортируем первым: ниже стандартные модули берём под псевдонимами,
# чтобы star-import движка не перекрыл socket/threading/time/random/math.
from ursina import *
from ursina import time as utime
from ursina.shader import Shader
from ursina.shaders.unlit_shader import unlit_shader
from panda3d.core import Filename, PerspectiveLens, Vec3 as PVec3

import argparse
import builtins
import json as pyjson
import math as pymath
import os as pyos
import queue as pyqueue
import random as pyrandom
import socket as pysocket
import sys as pysys
import tempfile
import threading as pthreading
import time as pytime


# ----------------------------------------------------------------------------
# КОНСТАНТЫ МИРА
# ----------------------------------------------------------------------------
SEED = 1313
WORLD_HALF = 72.0
GRID_STEP = 2.0
FOG_START = 2.0
FOG_END = 55.0
EYE_HEIGHT = 1.62
PLAYER_RADIUS = 0.45
DEFAULT_PORT = 7778
NET_RATE = 0.05          # 20 Гц — состояние игрока
MONSTER_NET_RATE = 0.066  # 15 Гц — состояние монстра
CATCH_RADIUS = 2.3


# ----------------------------------------------------------------------------
# МЕЛКИЕ МАТЕМАТИЧЕСКИЕ ХЕЛПЕРЫ
# ----------------------------------------------------------------------------
def clampf(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def clamp01(v):
    return clampf(v, 0.0, 1.0)


def lerpf(a, b, t):
    return a + (b - a) * t


def smoothstepf(a, b, x):
    if a == b:
        return 1.0 if x >= b else 0.0
    t = clamp01((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


def dist2d(a, b):
    return pymath.hypot(a.x - b.x, a.z - b.z)


def dist3d(a, b):
    return pymath.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def lerp_angle(a, b, t):
    """Плавный поворот по кратчайшей дуге (градусы)."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + d * t


# ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------
def terrain_height(x, z):
    """
    Процедурная высота сопки.
    Двор — почти ровный снег. Восточная сторона (x > 26) поднимается
    на 4.4 м каменным уступом. Плавный smoothstep-переход даёт
    естественную лестничную зону, а сами ступени строятся отдельно.
    """
    h = 0.11 * pymath.sin(x * 0.31) * pymath.cos(z * 0.27)
    h += 0.05 * pymath.sin(x * 0.83 + z * 0.61)

    # Снежный надув на юго-западе.
    h += 0.38 * pymath.exp(-(((x + 42.0) ** 2 + (z - 30.0) ** 2) / 190.0))
    # Небольшой бугор во дворе.
    h += 0.24 * pymath.exp(-(((x - 8.0) ** 2 + (z + 10.0) ** 2) / 130.0))

    # Каменная терраса на востоке.
    terrace = smoothstepf(26.0, 30.0, x)
    h += 4.4 * terrace

    return h


def terrain_normal(x, z):
    e = 0.15
    dx = terrain_height(x + e, z) - terrain_height(x - e, z)
    dz = terrain_height(x, z + e) - terrain_height(x, z - e)
    n = Vec3(-dx / (2.0 * e), 1.0, -dz / (2.0 * e))
    return n.normalized()


# ----------------------------------------------------------------------------
# ШЕЙДЕР ФОНАРИКА
# lit_with_shadows_shader в Ursina 8.3 игнорирует конус и затухание,
# поэтому используется собственный простой GLSL-шейдер: реальный SpotLight,
# затухание, мягкий край конуса и линейный туман.
# ----------------------------------------------------------------------------
FLASHLIGHT_SHADER = Shader(
    language=Shader.GLSL,
    name='skalnaya_flashlight_shader',
    vertex='''#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
in vec4 vertex;
in vec3 normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
uniform vec2 texture_scale;
uniform vec2 texture_offset;
out vec2 texcoords;
out vec4 vertex_color;
out vec3 v_view_pos;
out vec3 v_normal;
out vec3 v_world_pos;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    v_view_pos = (p3d_ModelViewMatrix * vertex).xyz;
    v_normal = normalize(p3d_NormalMatrix * normal);
    v_world_pos = (p3d_ModelMatrix * vertex).xyz;
    texcoords = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    vertex_color = p3d_Color;
}
''',
    fragment='''#version 150
uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform vec4 fog_color;
uniform float fog_start;
uniform float fog_end;
uniform vec3 camera_world_position;
uniform vec3 ambient_color;
in vec2 texcoords;
in vec4 vertex_color;
in vec3 v_view_pos;
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 fragment_color;
void main() {
    vec4 base = texture(p3d_Texture0, texcoords) * p3d_ColorScale * vertex_color;
    vec3 N = normalize(v_normal);
    vec4 lp = p3d_LightSource[0].position;
    vec3 diff = lp.xyz - v_view_pos * lp.w;
    float dist = length(diff);
    vec3 L = normalize(diff);
    float ndl = max(dot(N, L), 0.0);
    float att = 1.0 / max(
        p3d_LightSource[0].attenuation.x
        + p3d_LightSource[0].attenuation.y * dist
        + p3d_LightSource[0].attenuation.z * dist * dist,
        0.0001);
    float spot = 1.0;
    if (lp.w != 0.0) {
        float cosang = dot(-L, normalize(p3d_LightSource[0].spotDirection));
        spot = smoothstep(p3d_LightSource[0].spotCosCutoff,
                          p3d_LightSource[0].spotCosCutoff + 0.07,
                          cosang);
    }
    vec3 lit = base.rgb * (ambient_color
        + p3d_LightSource[0].color.rgb * ndl * att * spot);
    float dc = length(v_world_pos - camera_world_position);
    float t = clamp((dc - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
    lit = mix(lit, fog_color.rgb, t * fog_color.a);
    fragment_color = vec4(lit, base.a);
}
''',
    default_input={
        'texture_scale': Vec2(1, 1),
        'texture_offset': Vec2(0, 0),
        'fog_color': color.rgba32(7, 11, 20, 255),
        'fog_start': FOG_START,
        'fog_end': FOG_END,
        'camera_world_position': Vec3(0, 0, 0),
        'ambient_color': Vec3(0.090, 0.105, 0.150),
    },
)


def _camera_world_position():
    return camera.world_position


FLASHLIGHT_SHADER.continuous_input['camera_world_position'] = _camera_world_position


# ----------------------------------------------------------------------------
# ПРОЦЕДУРНЫЕ ТЕКСТУРЫ (Pillow, без внешних файлов)
# ----------------------------------------------------------------------------
def _pil_modules():
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
        return Image, ImageDraw, ImageFilter, ImageFont
    except Exception:
        return None


def make_noise_image(size, base, spread, tint, seed, streaks=0):
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, ImageFilter, _ = mods
    rng = pyrandom.Random(seed)
    img = Image.new('RGB', (size, size))
    px = []
    for _ in range(size * size):
        v = int(clampf(base + rng.randint(-spread, spread), 0, 255))
        px.append((
            int(clampf(v + tint[0], 0, 255)),
            int(clampf(v + tint[1], 0, 255)),
            int(clampf(v + tint[2], 0, 255)),
        ))
    img.putdata(px)
    draw = ImageDraw.Draw(img)
    for _ in range(streaks):
        x = rng.randint(0, size)
        y = rng.randint(0, size)
        draw.line(
            [(x, y), (x + rng.randint(-30, 30), y + rng.randint(15, 55))],
            fill=(min(255, base + 35), min(255, base + 38), min(255, base + 42)),
            width=1,
        )
    return img.filter(ImageFilter.GaussianBlur(0.45))


def make_concrete_image(size=256, base=140, seed=21, panel=False):
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, ImageFilter, _ = mods
    rng = pyrandom.Random(seed)
    img = make_noise_image(size, base, 14, (2, 3, 5), seed)
    draw = ImageDraw.Draw(img)
    step = 64 if panel else 96
    for y in range(0, size, step):
        draw.line([(0, y), (size, y)], fill=(62, 66, 74), width=2)
    for x in range(0, size, 128):
        draw.line([(x, 0), (x, size)], fill=(66, 70, 78), width=2)
    overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for _ in range(26):
        x = rng.randint(0, size)
        y = rng.randint(0, size)
        r = rng.randint(6, 34)
        od.ellipse([x - r, y - r, x + r, y + r], fill=(30, 32, 38, rng.randint(14, 40)))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    return img.filter(ImageFilter.GaussianBlur(0.4))


def make_rock_image(size=256):
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, ImageFilter, _ = mods
    rng = pyrandom.Random(404)
    img = make_noise_image(size, 58, 22, (0, 2, 5), 404)
    draw = ImageDraw.Draw(img)
    for _ in range(22):
        x = rng.randint(0, size)
        y = rng.randint(0, size)
        pts = [(x, y)]
        for _ in range(4):
            x += rng.randint(-28, 28)
            y += rng.randint(8, 34)
            pts.append((x, y))
        draw.line(pts, fill=(26, 28, 34), width=rng.randint(1, 3))
    return img.filter(ImageFilter.GaussianBlur(0.5))


def make_metal_image(size=256):
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, ImageFilter, _ = mods
    img = make_noise_image(size, 58, 9, (1, 2, 4), 909)
    draw = ImageDraw.Draw(img)
    for x in range(0, size, 16):
        draw.line([(x, 0), (x, size)], fill=(34, 36, 42), width=2)
        draw.line([(x + 4, 0), (x + 4, size)], fill=(78, 82, 90), width=1)
    return img.filter(ImageFilter.GaussianBlur(0.3))


def make_snow_image(size=256):
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, ImageFilter, _ = mods
    img = make_noise_image(size, 202, 18, (-14, -8, 8), 77, streaks=70)
    overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    rng = pyrandom.Random(78)
    for _ in range(90):
        x = rng.randint(0, size)
        y = rng.randint(0, size)
        r = rng.randint(2, 9)
        od.ellipse([x - r, y - r, x + r, y + r], fill=(235, 240, 250, 30))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    return img.filter(ImageFilter.GaussianBlur(0.35))


def make_glow_image(size=128):
    mods = _pil_modules()
    if not mods:
        return None
    Image, _, _, _ = mods
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    px = []
    half = size / 2.0
    for y in range(size):
        for x in range(size):
            d = pymath.hypot(x - half + 0.5, y - half + 0.5) / half
            a = int(255 * max(0.0, 1.0 - d) ** 2.2)
            px.append((255, 228, 168, a))
    img.putdata(px)
    return img


def make_vignette_image(size=256):
    mods = _pil_modules()
    if not mods:
        return None
    Image, _, _, _ = mods
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    px = []
    half = size / 2.0
    for y in range(size):
        for x in range(size):
            d = pymath.hypot(x - half + 0.5, y - half + 0.5) / half
            a = int(255 * clamp01((d - 0.48) / 0.52) ** 1.35)
            px.append((0, 0, 0, a))
    img.putdata(px)
    return img


def make_face_image(size=768):
    """Процедурное лицо Полярного Скитальца для скримера."""
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, ImageFilter, ImageFont = mods
    rng = pyrandom.Random(1313)
    img = Image.new('RGB', (size, size), (3, 3, 6))
    draw = ImageDraw.Draw(img)

    # Кровавое свечение на фоне.
    for i in range(28, 0, -1):
        r = int(390 * i / 28.0)
        shade = int(38 * (1.0 - i / 28.0) ** 1.5)
        draw.ellipse([size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r],
                     fill=(shade, 0, 0))

    # Бледное вытянутое лицо.
    draw.ellipse([170, 55, 598, 725], fill=(156, 156, 164))
    draw.ellipse([170, 55, 598, 725], outline=(78, 78, 88), width=7)
    draw.ellipse([200, 80, 568, 420], fill=(144, 144, 152))

    # Глазницы.
    draw.ellipse([232, 232, 352, 366], fill=(6, 6, 9))
    draw.ellipse([416, 232, 536, 366], fill=(6, 6, 9))
    draw.ellipse([252, 250, 332, 348], fill=(12, 8, 10))
    draw.ellipse([436, 250, 516, 348], fill=(12, 8, 10))
    # Зрачки.
    draw.ellipse([276, 286, 308, 326], fill=(224, 226, 230))
    draw.ellipse([460, 286, 492, 326], fill=(224, 226, 230))
    draw.ellipse([284, 296, 300, 316], fill=(10, 8, 12))
    draw.ellipse([468, 296, 484, 316], fill=(10, 8, 12))
    # Красные прожилки у глаз.
    for ex in (292, 476):
        for _ in range(10):
            draw.line(
                [(ex + rng.randint(-30, 30), 320 + rng.randint(-20, 20)),
                 (ex + rng.randint(-80, 80), 390 + rng.randint(-30, 80))],
                fill=(112, 12, 14), width=rng.randint(1, 2),
            )

    # Нос.
    draw.polygon([(384, 372), (352, 462), (416, 462)], fill=(110, 110, 120))
    draw.line([(384, 390), (384, 455)], fill=(78, 78, 88), width=3)

    # Рваный рот.
    mouth = [
        (300, 492), (340, 478), (384, 492), (428, 478), (468, 500),
        (452, 548), (468, 594), (438, 636), (404, 606), (384, 652),
        (360, 608), (326, 638), (300, 594), (314, 548),
    ]
    draw.polygon(mouth, fill=(4, 3, 5))
    # Зубы.
    for i in range(9):
        x = 316 + i * 17
        draw.polygon([(x, 500), (x + 12, 500), (x + 6, 528)], fill=(196, 194, 186))
    for i in range(8):
        x = 324 + i * 17
        draw.polygon([(x, 612), (x + 11, 612), (x + 5, 586)], fill=(178, 176, 170))
    # Трещины по лицу.
    for _ in range(18):
        x = rng.randint(210, 560)
        y = rng.randint(100, 690)
        pts = [(x, y)]
        for _ in range(3):
            x += rng.randint(-22, 22)
            y += rng.randint(10, 34)
            pts.append((x, y))
        draw.line(pts, fill=(70, 68, 78), width=rng.randint(1, 2))

    # Шум и лёгкое размытие.
    noise = Image.effect_noise((size, size), 26).convert('RGB')
    img = Image.blend(img, noise, 0.08)
    img = img.filter(ImageFilter.GaussianBlur(0.9))

    # Виньетка поверх.
    vig = make_vignette_image(size)
    if vig is not None:
        black = Image.new('RGBA', (size, size), (0, 0, 0, 255))
        black.putalpha(vig.getchannel('A'))
        img = Image.alpha_composite(img.convert('RGBA'), black).convert('RGB')
    return img


def make_sign_image():
    mods = _pil_modules()
    if not mods:
        return None
    Image, ImageDraw, _, ImageFont = mods
    img = Image.new('RGBA', (512, 128), (12, 14, 18, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, 509, 125], outline=(120, 124, 132, 255), width=3)
    font = None
    for path in (
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/DejaVuSans-Bold.ttf',
    ):
        try:
            font = ImageFont.truetype(path, 62)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    try:
        draw.text((256, 66), 'ул. Дежнева, 18', font=font, fill=(222, 226, 234, 255),
                  anchor='mm')
    except Exception:
        draw.text((40, 30), 'ул. Дежнева, 18', font=font, fill=(222, 226, 234, 255))
    return img


# ----------------------------------------------------------------------------
# СЕТЬ: TCP + newline-delimited JSON, два потока, без сторонних библиотек
# ----------------------------------------------------------------------------
class Network:
    def __init__(self):
        self.sock = None
        self.role = None
        self.connected = False
        self.rx = pyqueue.Queue()
        self._send_lock = pthreading.Lock()
        self._stop = pthreading.Event()
        self._reader = None
        self._listener = None
        self._initial_buffer = ''
        self.peer = None
        self.last_error = ''

    # ---- запуск -----------------------------------------------------------
    def host(self, port=DEFAULT_PORT, timeout=300.0):
        try:
            srv = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_STREAM)
            srv.setsockopt(pysocket.SOL_SOCKET, pysocket.SO_REUSEADDR, 1)
            srv.bind(('0.0.0.0', port))
            srv.listen(1)
            srv.settimeout(timeout)
            self._listener = srv
            print(f'[СЕТЬ] Сервер слушает 0.0.0.0:{port}')
            print('[СЕТЬ] IP для Radmin VPN:')
            for ip in local_ips():
                print(f'        {ip}:{port}')
            print('[СЕТЬ] Ожидание второго игрока... (Ctrl+C — отмена)')
            conn, addr = srv.accept()
            conn.settimeout(1.0)
            try:
                conn.setsockopt(pysocket.IPPROTO_TCP, pysocket.TCP_NODELAY, 1)
            except Exception:
                pass
            self.sock = conn
            self.role = 'host'
            self.connected = True
            self.peer = addr
            self._send_raw({'type': 'welcome', 'role': 'client', 'seed': SEED})
            self._start_reader()
            print(f'[СЕТЬ] Подключился клиент: {addr[0]}:{addr[1]}')
            return True
        except Exception as exc:
            self.last_error = str(exc)
            print(f'[СЕТЬ] Ошибка сервера: {exc}')
            self.close()
            return False

    def connect(self, ip, port=DEFAULT_PORT, timeout=20.0):
        deadline = pytime.time() + timeout
        last = None
        while pytime.time() < deadline and not self._stop.is_set():
            try:
                s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_STREAM)
                s.settimeout(4.0)
                s.connect((ip, port))
                try:
                    s.setsockopt(pysocket.IPPROTO_TCP, pysocket.TCP_NODELAY, 1)
                except Exception:
                    pass
                s.settimeout(1.0)
                self.sock = s
                self.role = 'client'
                self.peer = (ip, port)
                # Ждём welcome, чтобы синхронизировать старт.
                buf = ''
                welcome = None
                end = pytime.time() + 6.0
                while pytime.time() < end:
                    try:
                        data = s.recv(8192)
                    except pysocket.timeout:
                        continue
                    if not data:
                        break
                    buf += data.decode('utf-8', 'ignore')
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        if not line.strip():
                            continue
                        try:
                            msg = pyjson.loads(line)
                        except Exception:
                            continue
                        if msg.get('type') == 'welcome':
                            welcome = msg
                        else:
                            self.rx.put(msg)
                    if welcome:
                        break
                if not welcome:
                    raise RuntimeError('хост не прислал приветствие')
                self.connected = True
                self._initial_buffer = buf
                self._start_reader()
                print(f'[СЕТЬ] Подключено к {ip}:{port}')
                return True
            except Exception as exc:
                last = exc
                self.close()
                pytime.sleep(1.0)
        self.last_error = str(last) if last else 'таймаут'
        print(f'[СЕТЬ] Не удалось подключиться: {self.last_error}')
        return False

    # ---- обмен ------------------------------------------------------------
    def _start_reader(self):
        self._stop.clear()
        self._reader = pthreading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self):
        buf = self._initial_buffer
        self._initial_buffer = ''
        while not self._stop.is_set():
            try:
                data = self.sock.recv(8192)
            except pysocket.timeout:
                continue
            except Exception:
                break
            if not data:
                break
            buf += data.decode('utf-8', 'ignore')
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                if not line.strip():
                    continue
                try:
                    msg = pyjson.loads(line)
                except Exception:
                    continue
                self.rx.put(msg)
        if not self._stop.is_set():
            self.connected = False
            self.rx.put({'type': '_disconnected'})

    def _send_raw(self, obj):
        if not self.sock:
            return False
        data = (pyjson.dumps(obj, separators=(',', ':')) + '\n').encode('utf-8')
        try:
            with self._send_lock:
                self.sock.sendall(data)
            return True
        except Exception:
            self.connected = False
            return False

    def send(self, obj):
        if not self.connected:
            return False
        return self._send_raw(obj)

    def close(self):
        self._stop.set()
        self.connected = False
        for s in (self.sock, self._listener):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self.sock = None
        self._listener = None


def local_ips():
    ips = set()
    try:
        for ip in pysocket.gethostbyname_ex(pysocket.gethostname())[2]:
            if not ip.startswith('127.'):
                ips.add(ip)
    except Exception:
        pass
    try:
        s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        if ip and not ip.startswith('127.'):
            ips.add(ip)
        s.close()
    except Exception:
        pass
    return sorted(ips)


# ----------------------------------------------------------------------------
# ИГРА
# ----------------------------------------------------------------------------
class Game:
    def __init__(self, net=None, net_role='solo', smoke_test=False):
        self.net = net
        self.net_role = net_role
        self.net_id = 'host' if net_role == 'host' else ('client' if net_role == 'client' else 'solo')
        self.authority = net_role in ('host', 'solo')
        self.net_connected = bool(net and net.connected)
        self.smoke_test = smoke_test

        self.state = 'play'
        self.time_alive = 0.0
        self.fear = 0.0
        self.keys_collected = 0
        self.message_timer = 0.0
        self.net_timer = 0.0
        self.monster_net_timer = 0.0
        self.scare_t = 0.0
        self.win_t = 0.0
        self.game_over_victim = None

        self.remote = {
            'pos': None, 'rot': 0.0, 'flash': False,
            'sprint': False, 'moving': False, 'last': 0.0,
        }
        self.solids = []
        self.keys = []
        self.batteries = []
        self.snowflakes = []
        self.textures = {}

        self._setup_window()
        self._build_textures()
        self._build_world()

        self.player = Player(self)
        self.monster = Monster(self)
        self.monster.network_controlled = (not self.authority) and self.net_connected
        self.remote_player = RemotePlayer(self)

        self._build_ui()
        self._build_effects()

        scene.fog_color = color.rgba32(7, 11, 20, 255)
        scene.fog_density = (FOG_START, FOG_END)

        mouse.locked = True
        mouse.visible = False

        self._update_objective()
        self._show_message(
            'Роста. Полярная ночь. Найдите 5 ключей от старой машины.',
            9.0,
        )

        if smoke_test:
            invoke(self._smoke_normal, delay=1.2)
            invoke(self._smoke_scare, delay=3.0)
            invoke(self._smoke_shot, delay=6.0)
            invoke(application.quit, delay=8.0)

    # ---- окно -------------------------------------------------------------
    def _setup_window(self):
        window.title = 'Роста — Полярная ночь'
        window.borderless = False
        window.size = (1280, 720)
        window.color = color.rgb32(4, 6, 12)
        for attr, value in (('exit_button', None), ('fps_counter', False),
                            ('entity_counter', False), ('collider_counter', False),
                            ('cog_button', False)):
            obj = getattr(window, attr, None)
            if obj is None:
                continue
            try:
                if attr == 'exit_button':
                    obj.visible = False
                else:
                    obj.enabled = value
            except Exception:
                pass
        camera.fov = 82

    # ---- текстуры ---------------------------------------------------------
    def _tex(self, image, filtering='bilinear'):
        if image is None:
            return None
        try:
            return Texture(image, filtering=filtering)
        except Exception:
            return None

    def _build_textures(self):
        self.textures['snow'] = self._tex(make_snow_image(), 'mipmap')
        self.textures['concrete'] = self._tex(make_concrete_image())
        self.textures['panel'] = self._tex(make_concrete_image(panel=True, seed=33))
        self.textures['rock'] = self._tex(make_rock_image())
        self.textures['metal'] = self._tex(make_metal_image())
        self.textures['glow'] = self._tex(make_glow_image())
        self.textures['vignette'] = self._tex(make_vignette_image())
        self.textures['face'] = self._tex(make_face_image())
        self.textures['sign'] = self._tex(make_sign_image())

    # ---- геометрия --------------------------------------------------------
    def _add_solid(self, cx, cz, w, d, y0, y1, kind='building'):
        self.solids.append({
            'minx': cx - w / 2.0, 'maxx': cx + w / 2.0,
            'minz': cz - d / 2.0, 'maxz': cz + d / 2.0,
            'y0': y0, 'y1': y1, 'kind': kind,
        })

    def _box(self, cx, cy, cz, sx, sy, sz, col, tex=None,
             shader=None, rot=(0, 0, 0), solid=False, kind='building'):
        e = Entity(
            model='cube',
            position=(cx, cy, cz),
            scale=(sx, sy, sz),
            rotation=rot,
            color=col,
            texture=tex,
            shader=shader or FLASHLIGHT_SHADER,
        )
        if solid:
            self._add_solid(cx, cz, sx, sz, cy - sy / 2.0, cy + sy / 2.0, kind)
        return e

    def _quad(self, x, y, z, sx, sy, col, tex=None, billboard=False,
              unlit=True, rot=(0, 0, 0)):
        return Entity(
            model='quad',
            position=(x, y, z),
            scale=(sx, sy),
            rotation=rot,
            color=col,
            texture=tex,
            unlit=unlit,
            billboard=billboard,
            double_sided=True,
            shader=unlit_shader if unlit else FLASHLIGHT_SHADER,
        )

    # ---- мир --------------------------------------------------------------
    def _build_world(self):
        self._build_terrain()
        self._build_building()
        self._build_garages()
        self._build_fences()
        self._build_stairs()
        self._build_rocks()
        self._build_lamps()
        self._build_car()
        self._build_pickups()
        self._build_snow()
        self._build_skyline()

    def _build_terrain(self):
        """
        Сетка рельефа строится аналитически по terrain_height().
        Вершины окрашиваются: снег на пологих участках, тёмный камень
        на крутых склонах. Это даёт заснеженную сопку без внешних моделей.
        """
        n = int((WORLD_HALF * 2.0) / GRID_STEP)
        verts, tris, cols, uvs, norms = [], [], [], [], []
        for iz in range(n + 1):
            z = -WORLD_HALF + iz * GRID_STEP
            for ix in range(n + 1):
                x = -WORLD_HALF + ix * GRID_STEP
                h = terrain_height(x, z)
                nrm = terrain_normal(x, z)
                verts.append(Vec3(x, h, z))
                norms.append(nrm)
                uvs.append(Vec2(x / 7.0, z / 7.0))
                slope = nrm.y
                if slope > 0.80:
                    base = 188
                elif slope > 0.66:
                    base = 120
                else:
                    base = 62
                var = 0.86 + 0.14 * (0.5 + 0.5 * pymath.sin(x * 1.7 + z * 1.3))
                cols.append(color.rgb32(
                    int(clampf(base * var, 0, 255)),
                    int(clampf((base + 8) * var, 0, 255)),
                    int(clampf((base + 20) * var, 0, 255)),
                ))
        for iz in range(n):
            for ix in range(n):
                i0 = iz * (n + 1) + ix
                i1 = i0 + 1
                i2 = i0 + (n + 1)
                i3 = i2 + 1
                tris.extend((i0, i1, i2, i1, i3, i2))
        mesh = Mesh(vertices=verts, triangles=tris, colors=cols,
                    uvs=uvs, normals=norms, mode='triangle')
        self.terrain = Entity(
            model=mesh,
            texture=self.textures.get('snow'),
            shader=FLASHLIGHT_SHADER,
            color=color.white,
        )

    def _build_building(self):
        """Девятиэтажка росты: два блока и центральная арка."""
        panel = self.textures.get('panel')
        snow = self.textures.get('snow')
        wall = color.rgb32(196, 202, 214)
        dark = color.rgb32(150, 156, 168)

        # Левый и правый блоки, между ними проход-арка 4 м.
        self._box(-13.5, 15.0, -36.0, 23.0, 30.0, 12.0,
                  wall, panel, solid=True, kind='building')
        self._box(13.5, 15.0, -36.0, 23.0, 30.0, 12.0,
                  wall, panel, solid=True, kind='building')
        # Перемычка над аркой (проход снизу свободен).
        self._box(0.0, 17.5, -36.0, 4.0, 25.0, 12.0,
                  wall, panel, solid=True, kind='building')
        # Снег на крыше.
        self._box(0.0, 30.25, -36.0, 50.6, 0.35, 12.5,
                  color.rgb32(198, 208, 224), snow)

        # Окна: 8 колонок × 9 этажей на фасаде и сзади.
        rng = pyrandom.Random(SEED + 1)
        for cx in (-21, -16, -11, -6, 6, 11, 16, 21):
            for floor in range(9):
                y = 2.9 + floor * 3.0
                lit = rng.random() < 0.34
                wcol = (color.rgb32(255, 198, 112) if lit
                        else color.rgb32(13, 19, 31))
                for zf in (-29.92, -42.08):
                    self._quad(cx, y, zf, 1.25, 1.55, wcol)

        # Балконы.
        for cx in (-21, -11, 6, 16):
            for floor in (2, 5, 7):
                y = 2.9 + floor * 3.0 - 0.9
                self._box(cx, y, -29.55, 2.1, 0.16, 0.75,
                          color.rgb32(72, 76, 86), panel)
                self._box(cx, y + 0.48, -29.20, 2.1, 0.78, 0.06,
                          color.rgb32(52, 56, 66), panel)

        # Подъезды с козырьками и лампами.
        glow = self.textures.get('glow')
        for dx in (-16.0, 14.0):
            self._quad(dx, 1.15, -29.90, 1.7, 2.3,
                       color.rgb32(26, 22, 20))
            self._box(dx, 2.65, -29.55, 2.5, 0.18, 1.1,
                      color.rgb32(70, 74, 82), panel)
            self._quad(dx, 2.95, -29.35, 0.28, 0.14,
                       color.rgb32(255, 205, 120), tex=glow, billboard=False)
            self._quad(dx, 2.95, -29.15, 1.7, 1.7,
                       color.rgba32(255, 210, 130, 120), tex=glow,
                       billboard=True)

        # Табличка «ул. Дежнева, 18».
        sign = self.textures.get('sign')
        if sign:
            self._quad(0.0, 6.65, -29.86, 5.2, 1.3,
                       color.white, tex=sign, rot=(0, 180, 0))

        # Дуга арки на фасаде и сзади.
        for ang in range(0, 181, 20):
            rad = pymath.radians(ang)
            ax = 2.05 * pymath.cos(rad)
            ay = 3.0 + 2.05 * pymath.sin(rad)
            for zf in (-29.88, -42.12):
                self._box(ax, ay, zf, 0.55, 0.55, 0.35, dark, panel)
        for sx in (-2.05, 2.05):
            for zf in (-29.88, -42.12):
                self._box(sx, 1.5, zf, 0.5, 3.0, 0.35, dark, panel)

        # Лампа внутри арки.
        self._quad(0.0, 4.72, -36.0, 0.4, 0.2,
                   color.rgb32(255, 205, 120), tex=glow)
        self._quad(0.0, 4.72, -36.0, 2.6, 2.6,
                   color.rgba32(255, 205, 130, 100), tex=glow,
                   billboard=True)

    def _build_garages(self):
        """Тёмные советские гаражи-«кооперативы» на западе и за домом."""
        concrete = self.textures.get('concrete')
        metal = self.textures.get('metal')
        snow = self.textures.get('snow')
        wall = color.rgb32(176, 182, 194)
        door = color.rgb32(120, 126, 138)

        # Западный ряд: фасады смотрят на восток.
        for cz in (-10, 0, 10, 20, 30):
            self._box(-53.0, 1.6, cz, 10.0, 3.2, 8.0,
                      wall, concrete, solid=True, kind='garage')
            self._box(-53.0, 3.32, cz, 10.4, 0.24, 8.4,
                      color.rgb32(198, 208, 224), snow)
            self._quad(-47.96, 1.45, cz, 7.4, 2.6, door, tex=metal,
                       unlit=False, rot=(0, 90, 0))

        # Задний ряд: фасады смотрят во двор (+Z).
        for cx in (-18, -6, 6, 18):
            self._box(cx, 1.6, -50.0, 8.0, 3.2, 8.0,
                      wall, concrete, solid=True, kind='garage')
            self._box(cx, 3.32, -50.0, 8.4, 0.24, 8.4,
                      color.rgb32(198, 208, 224), snow)
            self._quad(cx, 1.45, -45.96, 6.9, 2.6, door, tex=metal,
                       unlit=False)

    def _build_fences(self):
        """Заснеженные бетонные заборы по периметру."""
        concrete = self.textures.get('concrete')
        snow = self.textures.get('snow')
        col = color.rgb32(186, 192, 204)
        snow_col = color.rgb32(198, 208, 224)

        def panel(cx, cz, along_x):
            sx = 6.2 if along_x else 0.4
            sz = 0.4 if along_x else 6.2
            self._box(cx, 1.3, cz, sx, 2.6, sz, col, concrete,
                      solid=True, kind='fence')
            self._box(cx, 2.66, cz, sx + 0.08, 0.18, sz + 0.08,
                      snow_col, snow)
            self._box(cx, 1.3, cz, 0.5 if along_x else 0.55, 2.8,
                      0.5 if along_x else 0.55, col, concrete)

        z = -60.0
        while z <= 60.0:
            panel(-66.0, z, False)
            panel(66.0, z, False)
            z += 6.5
        x = -60.0
        while x <= 60.0:
            panel(x, -66.0, True)
            panel(x, 66.0, True)
            x += 6.5

    def _build_stairs(self):
        """Бетонная лестница с двора на каменную террасу."""
        concrete = self.textures.get('concrete')
        col = color.rgb32(188, 194, 206)
        rail = color.rgb32(130, 136, 148)
        steps = 12
        for i in range(steps):
            top = 4.4 * (i + 1) / steps
            cx = 26.0 + (i + 0.5) * (4.0 / steps)
            self._box(cx, top / 2.0, 11.0, (4.0 / steps) + 0.03,
                      top, 10.0, col, concrete)
        # Перила вдоль лестницы.
        angle = -pymath.degrees(pymath.atan2(4.4, 4.0))
        for zz in (6.25, 15.75):
            self._box(28.0, 3.0, zz, 4.7, 0.09, 0.09, rail,
                      rot=(0, 0, angle))
            for px in (26.4, 29.6):
                self._box(px, terrain_height(px, zz) + 0.55, zz,
                          0.09, 1.1, 0.09, rail)

        # Невидимые скальные барьеры: подняться можно только по лестнице.
        self._add_solid(28.0, -30.5, 4.0, 71.0, 0.0, 8.0, kind='cliff')
        self._add_solid(28.0, 41.5, 4.0, 49.0, 0.0, 8.0, kind='cliff')

    def _build_rocks(self):
        """Скальные глыбы: сопка, валуны во дворе, снежные шапки."""
        rock_tex = self.textures.get('rock')
        snow_tex = self.textures.get('snow')
        rock_col = color.rgb32(158, 164, 176)
        snow_col = color.rgb32(196, 206, 222)
        rng = pyrandom.Random(SEED + 7)
        key_spots = [(-16, -27), (-62, 6), (40, -8), (0, -58), (-40, 34)]

        def forbidden(x, z):
            if -27 < x < 27 and -44 < z < -28:
                return True
            if -60 < x < -46 and -16 < z < 36:
                return True
            if -24 < x < 24 and -56 < z < -44:
                return True
            if pymath.hypot(x - 14, z - 40) < 8:
                return True
            if pymath.hypot(x, z - 8) < 14:
                return True
            if 24 < x < 32 and 4 < z < 18:
                return True
            for kx, kz in key_spots:
                if pymath.hypot(x - kx, z - kz) < 3.5:
                    return True
            return False

        for _ in range(58):
            x = z = 0.0
            ok = False
            for _ in range(24):
                x = rng.uniform(-63, 63)
                z = rng.uniform(-63, 63)
                if not forbidden(x, z):
                    ok = True
                    break
            if not ok:
                continue
            s = rng.uniform(0.50, 1.75)
            if rng.random() < 0.18:
                s *= 1.45
            y = terrain_height(x, z) + s * 0.22
            rot = (rng.uniform(-14, 14), rng.uniform(0, 360),
                   rng.uniform(-14, 14))
            self._box(x, y, z, s * 1.5, s, s * 1.25, rock_col, rock_tex,
                      rot=rot, solid=(s > 1.45), kind='rock')
            self._box(x, y + s * 0.48, z, s * 1.05, s * 0.18, s * 0.9,
                      snow_col, snow_tex, rot=rot)

        # коч.
        for z in list(range(-64, 5, 5)) + list(range(18, 65, 5)):
            y = terrain_height(28.0, z)
            s = 4.6
            self._box(28.0, y + 1.2, z, 5.2, s, 6.0, rock_col, rock_tex,
                      rot=(0, rng.uniform(-18, 18), rng.uniform(-6, 6)))
            self._box(28.0, y + 3.9, z, 4.2, 0.3, 4.8,
                      snow_col, snow_tex, rot=(0, rng.uniform(-12, 12), 0))

    def _build_lamps(self):
        """Тусклые дворовые фонари (эмиссивные, без реального света)."""
        glow = self.textures.get('glow')
        snow = self.textures.get('snow')
        pole_col = color.rgb32(96, 102, 114)
        lamp_col = color.rgb32(255, 204, 120)
        spots = [(-22, 22), (22, 16), (-48, 38), (38, 24),
                 (-34, -44), (14, 48), (-26, -58)]
        for x, z in spots:
            y = terrain_height(x, z)
            Entity(model=Cylinder(resolution=6, radius=0.08, height=4.6,
                                  start=0.0),
                   position=(x, y, z), color=pole_col,
                   shader=FLASHLIGHT_SHADER)
            self._box(x + 0.32, y + 4.5, z, 0.72, 0.08, 0.08, pole_col)
            self._box(x + 0.68, y + 4.4, z, 0.38, 0.18, 0.5, lamp_col,
                      shader=unlit_shader)
            self._box(x + 0.68, y + 4.53, z, 0.42, 0.08, 0.56,
                      color.rgb32(198, 208, 224), snow)
            self._quad(x + 0.68, y + 4.38, z, 2.9, 2.9,
                       color.rgba32(255, 210, 130, 110), tex=glow,
                       billboard=True)

    def _build_car(self):
        """Старый автомобиль — точка эвакуации."""
        y = terrain_height(14, 40)
        self.car_pos = Vec3(14, y, 40)
        self.car = Entity(position=(14, y, 40), rotation_y=90)
        body_col = color.rgb32(132, 154, 136)
        rust = color.rgb32(154, 132, 104)
        dark = color.rgb32(62, 66, 74)
        glass = color.rgb32(20, 28, 40)
        snow_col = color.rgb32(198, 208, 224)
        snow_tex = self.textures.get('snow')
        metal = self.textures.get('metal')

        def part(cx, cy, cz, sx, sy, sz, col, tex=None, rot=(0, 0, 0)):
            return Entity(parent=self.car, model='cube',
                          position=(cx, cy, cz), scale=(sx, sy, sz),
                          rotation=rot, color=col, texture=tex,
                          shader=FLASHLIGHT_SHADER)

        part(0, 0.72, 0, 4.2, 0.8, 1.9, body_col, metal)
        part(-0.25, 1.4, 0, 2.2, 0.75, 1.75, rust, metal)
        part(-0.25, 1.82, 0, 2.3, 0.12, 1.85, snow_col, snow_tex)
        # Стёкла.
        for zz, ry in ((0.885, 0), (-0.885, 0)):
            Entity(parent=self.car, model='quad', position=(-0.25, 1.45, zz),
                   rotation_y=ry, scale=(1.75, 0.55), color=glass,
                   unlit=True, double_sided=True, shader=unlit_shader)
        Entity(parent=self.car, model='quad', position=(0.87, 1.45, 0),
               rotation_y=90, scale=(1.4, 0.55), color=glass,
               unlit=True, double_sided=True, shader=unlit_shader)
        Entity(parent=self.car, model='quad', position=(-1.38, 1.45, 0),
               rotation_y=90, scale=(1.4, 0.55), color=glass,
               unlit=True, double_sided=True, shader=unlit_shader)
        # Колёса.
        for wx in (-1.35, 1.35):
            for wz in (-0.99, 0.99):
                Entity(parent=self.car,
                       model=Cylinder(resolution=10, radius=0.38,
                                      height=0.3, start=-0.15),
                       position=(wx, 0.38, wz), rotation=(90, 0, 0),
                       color=dark, shader=FLASHLIGHT_SHADER)
                Entity(parent=self.car,
                       model=Cylinder(resolution=8, radius=0.17,
                                      height=0.32, start=-0.16),
                       position=(wx, 0.38, wz), rotation=(90, 0, 0),
                       color=color.rgb32(96, 100, 108),
                       shader=FLASHLIGHT_SHADER)
        # Фары.
        glow = self.textures.get('glow')
        for wz in (-0.62, 0.62):
            Entity(parent=self.car, model='cube',
                   position=(2.12, 0.76, wz), scale=(0.08, 0.22, 0.3),
                   color=color.rgb32(255, 224, 150), unlit=True,
                   shader=unlit_shader)
            Entity(parent=self.car, model='quad',
                   position=(2.18, 0.76, wz), scale=(0.9, 0.9),
                   texture=glow, color=color.rgba32(255, 210, 120, 90),
                   billboard=True, unlit=True, shader=unlit_shader)
        self._add_solid(14, 40, 2.3, 4.6, y, y + 2.1, kind='car')

    def _build_pickups(self):
        """5 ключей и 4 батарейки. Позиции одинаковы у хоста и клиента."""
        key_tex = None
        glow = self.textures.get('glow')
        key_col = color.rgb32(238, 194, 72)
        key_spots = [(-16, -27), (-62, 6), (40, -8), (0, -58), (-40, 34)]
        for i, (x, z) in enumerate(key_spots):
            base_y = terrain_height(x, z)
            root = Entity(position=(x, base_y + 1.0, z))
            for j in range(8):
                a = 2.0 * pymath.pi * j / 8.0
                Entity(parent=root, model='cube',
                       position=(pymath.cos(a) * 0.11,
                                 pymath.sin(a) * 0.11, 0),
                       scale=(0.045, 0.045, 0.035), color=key_col,
                       unlit=True, shader=unlit_shader)
            Entity(parent=root, model='cube', position=(0, -0.2, 0),
                   scale=(0.045, 0.3, 0.035), color=key_col,
                   unlit=True, shader=unlit_shader)
            Entity(parent=root, model='cube', position=(0.07, -0.31, 0),
                   scale=(0.1, 0.045, 0.035), color=key_col,
                   unlit=True, shader=unlit_shader)
            Entity(parent=root, model='cube', position=(0.06, -0.23, 0),
                   scale=(0.08, 0.04, 0.035), color=key_col,
                   unlit=True, shader=unlit_shader)
            Entity(parent=root, model='quad', position=(0, 0, 0),
                   scale=(1.0, 1.0), texture=glow,
                   color=color.rgba32(255, 205, 80, 80), billboard=True,
                   unlit=True, shader=unlit_shader)
            self.keys.append({
                'pos': Vec3(x, base_y, z),
                'root': root,
                'base_y': base_y + 1.0,
                'phase': i * 1.7,
                'collected': False,
            })

        bat_spots = [(-30, 12), (22, 30), (34, 14), (-52, -30)]
        bat_col = color.rgb32(36, 42, 54)
        band_col = color.rgb32(228, 196, 64)
        for i, (x, z) in enumerate(bat_spots):
            base_y = terrain_height(x, z)
            root = Entity(position=(x, base_y + 0.75, z))
            Entity(parent=root, model='cube', position=(0, 0, 0),
                   scale=(0.18, 0.32, 0.18), color=bat_col,
                   unlit=True, shader=unlit_shader)
            Entity(parent=root, model='cube', position=(0, 0.02, 0),
                   scale=(0.19, 0.08, 0.19), color=band_col,
                   unlit=True, shader=unlit_shader)
            Entity(parent=root, model='cube', position=(0, 0.2, 0),
                   scale=(0.08, 0.07, 0.08),
                   color=color.rgb32(170, 176, 188),
                   unlit=True, shader=unlit_shader)
            Entity(parent=root, model='quad', position=(0, 0, 0),
                   scale=(0.9, 0.9), texture=glow,
                   color=color.rgba32(110, 220, 255, 65), billboard=True,
                   unlit=True, shader=unlit_shader)
            self.batteries.append({
                'pos': Vec3(x, base_y, z),
                'root': root,
                'base_y': base_y + 0.75,
                'phase': i * 2.1,
                'taken': False,
            })

    def _build_snow(self):
        """Метель: 220 снежинок вокруг игрока, с ветром и зацикливанием."""
        rng = pyrandom.Random(SEED + 11)
        for _ in range(220):
            e = Entity(model='cube', scale=(0.035, 0.035, 0.17),
                       color=color.rgba32(212, 222, 242, 210),
                       unlit=True, shader=unlit_shader)
            e.position = Vec3(
                rng.uniform(-22, 22),
                rng.uniform(0, 16),
                rng.uniform(-22, 22),
            )
            e.rotation = Vec3(0, rng.uniform(0, 360), 0)
            self.snowflakes.append({
                'e': e,
                'speed': rng.uniform(2.2, 4.8),
                'phase': rng.uniform(0, 6.28),
                'drift': rng.uniform(0.4, 1.1),
            })

    def _build_skyline(self):
        """Сугробы и дальние силуэты панелек за туманом."""
        snow_tex = self.textures.get('snow')
        snow_col = color.rgb32(196, 206, 224)
        rng = pyrandom.Random(SEED + 17)
        for _ in range(30):
            x = rng.uniform(-64, 64)
            z = rng.uniform(-64, 64)
            if -27 < x < 27 and -44 < z < -28:
                continue
            if pymath.hypot(x, z - 8) < 10:
                continue
            if pymath.hypot(x - 14, z - 40) < 5:
                continue
            sx = rng.uniform(1.8, 4.6)
            sy = rng.uniform(0.4, 1.0)
            sz = rng.uniform(1.8, 4.6)
            Entity(model='sphere', position=(x, terrain_height(x, z) + sy * 0.1, z),
                   scale=(sx, sy, sz), color=snow_col, texture=snow_tex,
                   shader=FLASHLIGHT_SHADER)

        # Дальние панельки — тёмные силуэты, почти съеденные туманом.
        for x, z, sx, sz, h in (
            (-85, -80, 34, 18, 34), (10, -95, 40, 20, 40),
            (90, -70, 30, 18, 32), (-95, 30, 28, 16, 30),
        ):
            Entity(model='cube', position=(x, h / 2.0, z),
                   scale=(sx, h, sz), color=color.rgb32(34, 38, 48),
                   shader=FLASHLIGHT_SHADER)

# ----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНАЯ ГЕОМЕТРИЯ
# ----------------------------------------------------------------------------
def make_cone_mesh(radius=1.5, length=9.0, segments=18):
    verts = [Vec3(0, 0, 0)]
    for i in range(segments):
        a = 2.0 * pymath.pi * i / segments
        verts.append(Vec3(pymath.cos(a) * radius,
                          pymath.sin(a) * radius, length))
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.extend((0, 1 + i, 1 + j))
    return Mesh(vertices=verts, triangles=tris, mode='triangle')


# ----------------------------------------------------------------------------
# ИГРОК ОТ ПЕРВОГО ЛИЦА
# ----------------------------------------------------------------------------
class Player:
    def __init__(self, game):
        self.game = game
        self.pos = Vec3(0, terrain_height(0, 8), 8)
        self.yaw = 180.0
        self.pitch = 0.0
        self.cam_y = self.pos.y + EYE_HEIGHT
        self.battery = 100.0
        self.stamina = 100.0
        self.flashlight_on = True
        self.moving = False
        self.sprinting = False
        self.bob_t = 0.0
        self.flicker_t = 0.0
        self.bob_y = 0.0
        self.roll = 0.0

        # Настоящий SpotLight, прикреплённый к камере.
        self.flashlight = SpotLight(
            parent=camera,
            position=(0.10, -0.10, 0.12),
            rotation=(3, 0, 0),
            color=color.rgb32(255, 242, 205),
        )
        try:
            lens = PerspectiveLens()
            lens.setFov(47)
            lens.setNearFar(0.1, 60)
            self.flashlight._light.setLens(lens)
            self.flashlight._light.setAttenuation(PVec3(1.0, 0.022, 0.0016))
        except Exception:
            pass

        # Объёмный конус луча убран: изнутри он даёт заметный круг на экране.
        # Свет даёт сам SpotLight, а не геометрия.
        self.beam = None

        self.reset()

    def reset(self):
        self.pos = Vec3(0, terrain_height(0, 8), 8)
        self.yaw = 180.0
        self.pitch = 0.0
        self.cam_y = self.pos.y + EYE_HEIGHT
        self.battery = 100.0
        self.stamina = 100.0
        self.flashlight_on = True
        self.moving = False
        self.sprinting = False
        self.bob_t = 0.0
        self.bob_y = 0.0
        self.roll = 0.0
        self.flashlight.enabled = True
        camera.position = (self.pos.x, self.cam_y, self.pos.z)
        camera.rotation = (0, self.yaw, 0)
        camera.fov = 82

    # ---- управление -------------------------------------------------------
    def toggle_flashlight(self):
        if self.game.state != 'play':
            return
        if self.battery <= 0.0:
            self.game._show_message('Батарейка села. Найдите запасную.', 3.0)
            return
        self.flashlight_on = not self.flashlight_on
        self.flashlight.enabled = self.flashlight_on

    def add_battery(self, amount):
        self.battery = clampf(self.battery + amount, 0.0, 100.0)
        if self.battery > 0.0:
            self.flashlight_on = True

    def update(self, dt):
        if self.game.state != 'play':
            return

        # Мышь.
        if mouse.locked:
            self.yaw += mouse.velocity[0] * 40.0
            self.pitch = clampf(self.pitch - mouse.velocity[1] * 40.0, -86.0, 86.0)

        fw = held_keys['w'] - held_keys['s']
        st = held_keys['d'] - held_keys['a']
        shift = bool(held_keys['left shift'] or held_keys['right shift'])

        move = Vec3(0, 0, 0)
        if fw or st:
            sy = pymath.sin(pymath.radians(self.yaw))
            cy = pymath.cos(pymath.radians(self.yaw))
            forward = Vec3(sy, 0, cy)
            right = Vec3(cy, 0, -sy)
            move = (forward * fw + right * st).normalized()

        self.moving = move.length() > 0.01
        self.sprinting = (self.moving and shift and self.stamina > 1.0)

        # Выносливость.
        if self.sprinting:
            self.stamina = max(0.0, self.stamina - 21.0 * dt)
        else:
            self.stamina = min(100.0, self.stamina + 13.0 * dt)

        # Фонарик разряжается.
        if self.flashlight_on:
            self.battery = max(0.0, self.battery - 0.30 * dt)
            if self.battery <= 0.0:
                self.flashlight_on = False
                self.flashlight.enabled = False
                self.game._show_message('Фонарик погас. Найдите батарейку.', 4.0)

        # Низкий заряд — мерцание.
        lit = self.flashlight_on and self.battery > 0.0
        if lit and self.battery < 18.0:
            self.flicker_t -= dt
            if self.flicker_t <= 0.0:
                self.flicker_t = pyrandom.uniform(0.05, 0.22)
                lit = pyrandom.random() > 0.18
        self.flashlight.enabled = lit

        # Движение с коллизиями (по осям — скольжение вдоль стен).
        if self.moving:
            speed = 6.4 if self.sprinting else 3.6
            step = speed * dt
            dx = move.x * step
            dz = move.z * step
            feet = terrain_height(self.pos.x, self.pos.z)
            if not self.game.player_blocked(self.pos.x + dx, self.pos.z, feet):
                self.pos.x += dx
            if not self.game.player_blocked(self.pos.x, self.pos.z + dz, feet):
                self.pos.z += dz
            self.pos.x = clampf(self.pos.x, -66.0, 66.0)
            self.pos.z = clampf(self.pos.z, -66.0, 66.0)

        target_y = terrain_height(self.pos.x, self.pos.z)
        self.pos.y = lerpf(self.pos.y, target_y, min(1.0, dt * 14.0))

        # Покачивание камеры.
        if self.moving:
            self.bob_t += dt * (10.5 if self.sprinting else 6.5)
        else:
            self.bob_t += dt * 2.0
        target_bob = (pymath.sin(self.bob_t) * (0.055 if self.sprinting else 0.035)
                      if self.moving else 0.0)
        target_roll = (pymath.sin(self.bob_t * 0.5) * 0.55
                       if self.moving else 0.0)
        self.bob_y = lerpf(self.bob_y, target_bob, min(1.0, dt * 10.0))
        self.roll = lerpf(self.roll, target_roll, min(1.0, dt * 8.0))

        shake = 0.0
        if self.game.fear > 0.55:
            shake = (self.game.fear - 0.55) * 0.045
        sx = pyrandom.uniform(-shake, shake)
        sy2 = pyrandom.uniform(-shake, shake)
        camera.position = (self.pos.x + sx,
                           self.pos.y + EYE_HEIGHT + self.bob_y + sy2,
                           self.pos.z)
        camera.rotation = (
            self.pitch + pyrandom.uniform(-shake, shake) * 35.0,
            self.yaw + pyrandom.uniform(-shake, shake) * 35.0,
            self.roll,
        )
        target_fov = 88 if self.sprinting else 82
        camera.fov = lerpf(camera.fov, target_fov, min(1.0, dt * 4.0))


# ----------------------------------------------------------------------------
# ВТОРОЙ ИГРОК (сетевой манекен)
# ----------------------------------------------------------------------------
class RemotePlayer:
    def __init__(self, game):
        self.game = game
        self.root = Entity(position=(0, -80, 0), enabled=False)
        body_col = color.rgb32(108, 116, 138)
        head_col = color.rgb32(140, 148, 168)
        dark = color.rgb32(78, 84, 102)

        self.body = Entity(parent=self.root,
                           model=Capsule(height=0.95, radius=0.30),
                           position=(0, 1.0, 0), color=body_col,
                           shader=FLASHLIGHT_SHADER)
        self.head = Entity(parent=self.root, model='sphere',
                           position=(0, 1.78, 0),
                           scale=(0.42, 0.46, 0.42), color=head_col,
                           shader=FLASHLIGHT_SHADER)
        Entity(parent=self.root, model='sphere',
               position=(0, 1.84, -0.03),
               scale=(0.50, 0.44, 0.48), color=dark,
               shader=FLASHLIGHT_SHADER)
        self.arm_l = Entity(parent=self.root, model='cube',
                            position=(-0.39, 1.20, 0),
                            scale=(0.15, 0.68, 0.15), color=dark,
                            rotation_z=7, shader=FLASHLIGHT_SHADER)
        self.arm_r = Entity(parent=self.root, model='cube',
                            position=(0.39, 1.20, 0),
                            scale=(0.15, 0.68, 0.15), color=dark,
                            rotation_z=-7, shader=FLASHLIGHT_SHADER)
        self.leg_l = Entity(parent=self.root, model='cube',
                            position=(-0.16, 0.35, 0),
                            scale=(0.18, 0.62, 0.20), color=dark,
                            shader=FLASHLIGHT_SHADER)
        self.leg_r = Entity(parent=self.root, model='cube',
                            position=(0.16, 0.35, 0),
                            scale=(0.18, 0.62, 0.20), color=dark,
                            shader=FLASHLIGHT_SHADER)
        # Фонарик на груди.
        Entity(parent=self.root, model='cube',
               position=(0.28, 1.24, 0.26),
               scale=(0.12, 0.12, 0.22),
               color=color.rgb32(255, 224, 140), unlit=True,
               shader=unlit_shader)
        self.beam = Entity(parent=self.root, model='quad',
                           position=(0.28, 1.24, 0.40),
                           scale=(0.55, 0.55),
                           texture=game.textures.get('glow'),
                           color=color.rgba32(255, 236, 175, 160),
                           billboard=True, unlit=True,
                           shader=unlit_shader, enabled=False)
        try:
            self.beam.set_depth_write(False)
        except Exception:
            pass

        self.target_pos = Vec3(0, -80, 0)
        self.target_yaw = 0.0
        self.has_state = False
        self.flash_on = False
        self.moving = False
        self.sprinting = False
        self.anim_t = 0.0

    def apply_state(self, x, y, z, ry, flash, moving, sprinting):
        self.target_pos = Vec3(x, y, z)
        self.target_yaw = ry
        self.flash_on = bool(flash)
        self.moving = bool(moving)
        self.sprinting = bool(sprinting)
        self.has_state = True
        if self.game.net_connected:
            self.root.enabled = True

    def reset(self):
        self.has_state = False
        self.root.enabled = False
        self.target_pos = Vec3(0, -80, 0)

    def update(self, dt):
        if not self.has_state or not self.game.net_connected:
            self.root.enabled = False
            return
        self.root.enabled = True
        t = min(1.0, dt * 12.0)
        self.root.x = lerpf(self.root.x, self.target_pos.x, t)
        self.root.y = lerpf(self.root.y, self.target_pos.y, t)
        self.root.z = lerpf(self.root.z, self.target_pos.z, t)
        self.root.rotation_y = lerp_angle(self.root.rotation_y,
                                          self.target_yaw, t)
        self.beam.enabled = self.flash_on
        if self.moving:
            self.anim_t += dt * (10.0 if self.sprinting else 7.0)
        swing = pymath.sin(self.anim_t) * (26.0 if self.moving else 0.0)
        self.arm_l.rotation_x = lerpf(self.arm_l.rotation_x, swing, t)
        self.arm_r.rotation_x = lerpf(self.arm_r.rotation_x, -swing, t)
        self.leg_l.rotation_x = lerpf(self.leg_l.rotation_x, -swing * 0.8, t)
        self.leg_r.rotation_x = lerpf(self.leg_r.rotation_x, swing * 0.8, t)
        bob = pymath.sin(self.anim_t * 2.0) * 0.025 if self.moving else 0.0
        self.body.y = 1.0 + bob
        self.head.y = 1.78 + bob


# ----------------------------------------------------------------------------
# ПОЛЯРНЫЙ СКИТАЛЕЦ — ИИ
# ----------------------------------------------------------------------------
class Monster:
    def __init__(self, game):
        self.game = game
        self.pos = Vec3(-40, terrain_height(-40, -52), -52)
        self.yaw = 0.0
        self.mode = 'patrol'
        self.target = Vec3(self.pos.x, self.pos.z, 0)
        self.waypoint_index = 1
        self.alert_timer = 0.0
        self.investigate_timer = 0.0
        self.last_known = Vec3(self.pos.x, self.pos.z, 0)
        self.chase_target = None
        self.anim_t = 0.0
        self.stuck_timer = 0.0
        self.network_controlled = False
        self.network_target = None
        self.network_yaw = 0.0
        self.network_mode = 'patrol'

        self.waypoints = [
            (-40, -52), (-20, -36), (10, -36), (22, -18), (10, 0),
            (-15, 8), (-38, 18), (-30, 30), (-5, 34), (20, 32),
            (10, 10), (-20, -10), (0, -20), (40, 5),
        ]

        body_col = color.rgb32(18, 20, 28)
        coat_col = color.rgb32(12, 14, 20)
        head_col = color.rgb32(22, 24, 32)
        self.root = Entity(position=(self.pos.x, self.pos.y, self.pos.z))
        self.coat = Entity(parent=self.root,
                           model=Cone(resolution=8, radius=0.62,
                                      height=1.6, add_bottom=False),
                           position=(0, 0.85, 0), color=coat_col,
                           unlit=True)
        self.body = Entity(parent=self.root,
                           model=Capsule(height=0.9, radius=0.28),
                           position=(0, 1.55, 0), color=body_col,
                           unlit=True)
        self.head = Entity(parent=self.root, model='sphere',
                           position=(0, 2.38, 0),
                           scale=(0.44, 0.60, 0.44), color=head_col,
                           unlit=True)
        Entity(parent=self.root, model='sphere',
               position=(0, 2.42, -0.04),
               scale=(0.54, 0.62, 0.50), color=coat_col,
               unlit=True)
        self.arm_l = Entity(parent=self.root, model='cube',
                            position=(-0.43, 1.45, 0),
                            scale=(0.13, 1.30, 0.13), color=body_col,
                            rotation_z=8, unlit=True)
        self.arm_r = Entity(parent=self.root, model='cube',
                            position=(0.43, 1.45, 0),
                            scale=(0.13, 1.30, 0.13), color=body_col,
                            rotation_z=-8, unlit=True)
        Entity(parent=self.root, model='sphere',
               position=(-0.49, 0.80, 0),
               scale=(0.15, 0.15, 0.15), color=body_col, unlit=True)
        Entity(parent=self.root, model='sphere',
               position=(0.49, 0.80, 0),
               scale=(0.15, 0.15, 0.15), color=body_col, unlit=True)
        Entity(parent=self.root, model='cube',
               position=(-0.17, 0.45, 0), scale=(0.16, 0.85, 0.18),
               color=coat_col, unlit=True)
        Entity(parent=self.root, model='cube',
               position=(0.17, 0.45, 0), scale=(0.16, 0.85, 0.18),
               color=coat_col, unlit=True)
        self.eye_l = Entity(parent=self.root, model='sphere',
                            position=(-0.11, 2.43, 0.20),
                            scale=(0.07, 0.07, 0.07),
                            color=color.rgb32(170, 230, 255),
                            unlit=True, shader=unlit_shader)
        self.eye_r = Entity(parent=self.root, model='sphere',
                            position=(0.11, 2.43, 0.20),
                            scale=(0.07, 0.07, 0.07),
                            color=color.rgb32(170, 230, 255),
                            unlit=True, shader=unlit_shader)
        glow = game.textures.get('glow')
        self.eye_glow = Entity(parent=self.root, model='quad',
                               position=(0, 2.42, 0.16),
                               scale=(0.75, 0.75), texture=glow,
                               color=color.rgba32(140, 215, 255, 55),
                               billboard=True, unlit=True,
                               shader=unlit_shader)
        self.chest_glow = Entity(parent=self.root, model='quad',
                                 position=(0, 1.45, 0.1),
                                 scale=(1.5, 1.5), texture=glow,
                                 color=color.rgba32(80, 130, 220, 26),
                                 billboard=True, unlit=True,
                                 shader=unlit_shader)
        self.reset()

    def reset(self):
        self.pos = Vec3(-40, terrain_height(-40, -52), -52)
        self.yaw = 0.0
        self.mode = 'patrol'
        self.waypoint_index = 1
        self.alert_timer = 0.0
        self.investigate_timer = 0.0
        self.last_known = Vec3(self.pos.x, self.pos.z, 0)
        self.chase_target = None
        self.stuck_timer = 0.0
        self.network_target = None
        self.network_yaw = 0.0
        self.network_mode = 'patrol'
        self.root.position = (self.pos.x, self.pos.y, self.pos.z)
        self.root.rotation_y = 0.0
        self._set_eyes(False)

    def apply_network_state(self, x, y, z, ry, mode):
        self.network_target = Vec3(x, y, z)
        self.network_yaw = ry
        self.network_mode = mode

    def _set_eyes(self, chasing):
        if chasing:
            c = color.rgb32(255, 46, 34)
            gc = color.rgba32(255, 40, 30, 90)
        else:
            c = color.rgb32(170, 230, 255)
            gc = color.rgba32(140, 215, 255, 55)
        self.eye_l.color = c
        self.eye_r.color = c
        self.eye_glow.color = gc

    def _animate(self, dt):
        chasing = self.mode == 'chase'
        self.anim_t += dt * (8.0 if chasing else 2.6)
        s = pymath.sin(self.anim_t)
        self.body.y = 1.55 + s * (0.045 if chasing else 0.02)
        self.head.y = 2.38 + s * (0.04 if chasing else 0.02)
        self.arm_l.rotation_x = s * (24.0 if chasing else 6.0)
        self.arm_r.rotation_x = -s * (24.0 if chasing else 6.0)
        lean = 10.0 if chasing else 0.0
        self.body.rotation_x = lerpf(self.body.rotation_x, lean, min(1.0, dt * 3.0))
        self.head.rotation_x = lerpf(self.head.rotation_x, lean * 0.5, min(1.0, dt * 3.0))
        self._set_eyes(chasing)

    def update(self, dt, targets):
        if self.network_controlled:
            if self.network_target is not None:
                t = min(1.0, dt * 9.0)
                self.pos.x = lerpf(self.pos.x, self.network_target.x, t)
                self.pos.y = lerpf(self.pos.y, self.network_target.y, t)
                self.pos.z = lerpf(self.pos.z, self.network_target.z, t)
                self.yaw = lerp_angle(self.yaw, self.network_yaw, t)
                self.mode = self.network_mode
            self.root.position = (self.pos.x, self.pos.y, self.pos.z)
            self.root.rotation_y = self.yaw
            self._animate(dt)
            return

        # --- ВОСПРИЯТИЕ -----------------------------------------------------
        detected = None
        best_score = -999.0
        for t in targets:
            d = dist2d(self.pos, t['pos'])
            if d > 34.0:
                continue
            blocked = self.game.line_blocked(self.pos, t['pos'])
            heard_radius = 20.0 if (t['sprinting'] and t['moving']) else (
                7.0 if t['moving'] else 3.0)
            if blocked:
                heard_radius *= 0.55
            heard = d < heard_radius
            seen = bool(t['flashlight']) and d < 30.0 and not blocked
            score = -999.0
            if seen:
                score = 200.0 - d
            elif heard:
                score = 100.0 - d
            elif d < 5.0:
                score = 50.0 - d
            if score > best_score:
                best_score = score
                detected = t

        if detected is not None:
            self.mode = 'chase'
            self.chase_target = detected['id']
            self.last_known = Vec3(detected['pos'].x, detected['pos'].z, 0)
            self.alert_timer = 9.0

        # --- СОСТОЯНИЯ ------------------------------------------------------
        if self.mode == 'chase':
            if detected is None:
                self.alert_timer -= dt
                if self.alert_timer <= 0.0:
                    self.mode = 'investigate'
                    self.investigate_timer = 5.0
            tx, tz = self.last_known.x, self.last_known.z
            speed = 4.7
        elif self.mode == 'investigate':
            self.investigate_timer -= dt
            tx, tz = self.last_known.x, self.last_known.z
            speed = 2.5
            if (self.investigate_timer <= 0.0
                    or pymath.hypot(self.pos.x - tx, self.pos.z - tz) < 1.5):
                self.mode = 'patrol'
                self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
                tx, tz = self.waypoints[self.waypoint_index]
        else:
            self.mode = 'patrol'
            wx, wz = self.waypoints[self.waypoint_index]
            tx, tz = wx, wz
            speed = 1.7
            if pymath.hypot(self.pos.x - tx, self.pos.z - tz) < 2.0:
                self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
                wx, wz = self.waypoints[self.waypoint_index]
                tx, tz = wx, wz

        # --- ДВИЖЕНИЕ С КОЛЛИЗИЯМИ -----------------------------------------
        dx = tx - self.pos.x
        dz = tz - self.pos.z
        d = pymath.hypot(dx, dz)
        if d > 0.05:
            step = min(speed * dt, d)
            ux, uz = dx / d, dz / d
            nx = self.pos.x + ux * step
            nz = self.pos.z + uz * step
            moved = False
            if not self.game.monster_blocked(nx, self.pos.z):
                self.pos.x = nx
                moved = True
            if not self.game.monster_blocked(self.pos.x, nz):
                self.pos.z = nz
                moved = True
            if not moved:
                # Обход угла: пробуем перпендикуляр.
                px, pz = -uz * step, ux * step
                if not self.game.monster_blocked(self.pos.x + px, self.pos.z + pz):
                    self.pos.x += px
                    self.pos.z += pz
                    moved = True
            self.stuck_timer = 0.0 if moved else self.stuck_timer + dt
            if self.stuck_timer > 1.4:
                self.stuck_timer = 0.0
                if self.mode == 'patrol':
                    self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
                else:
                    self.last_known.x += pyrandom.uniform(-4.0, 4.0)
                    self.last_known.z += pyrandom.uniform(-4.0, 4.0)

            target_yaw = pymath.degrees(pymath.atan2(ux, uz))
            self.yaw = lerp_angle(self.yaw, target_yaw, min(1.0, dt * 6.0))

        self.pos.y = terrain_height(self.pos.x, self.pos.z)
        self.root.position = (self.pos.x, self.pos.y, self.pos.z)
        self.root.rotation_y = self.yaw
        self._animate(dt)

# ----------------------------------------------------------------------------
# МЕТОДЫ ИГРЫ (UI, обновление, коллизии, сеть)
# ----------------------------------------------------------------------------
def _game_build_ui(self):
    self.vignette = None
    self.battery_fill, self.battery_label = self._hud_bar(
        -0.55, -0.405, 0.38, 0.022, 'ФОНАРЬ 100%', color.rgb32(110, 200, 255))
    self.battery_w = 0.38
    self.stamina_fill, self.stamina_label = self._hud_bar(
        -0.55, -0.452, 0.38, 0.022, 'ВЫНОСЛИВОСТЬ', color.rgb32(90, 220, 170))
    self.stamina_w = 0.38
    self.fear_fill, self.fear_label = self._hud_bar(
        0.0, -0.405, 0.36, 0.022, 'СТРАХ', color.rgb32(255, 220, 80))
    self.fear_w = 0.36

    self.objective_text = Text(
        parent=camera.ui, text='', position=(-0.86, 0.455),
        origin=(-0.5, 0), scale=1.05, color=color.rgb32(215, 222, 235),
        z=0.0,
    )
    self.keys_text = Text(
        parent=camera.ui, text='КЛЮЧИ 0/5', position=(0.86, 0.455),
        origin=(0.5, 0), scale=1.25, color=color.rgb32(238, 200, 90),
        z=0.0,
    )
    self.role_text = Text(
        parent=camera.ui, text='', position=(0.86, -0.468),
        origin=(0.5, 0), scale=0.8, color=color.rgb32(140, 150, 170),
        z=0.0,
    )
    self.prompt_text = Text(
        parent=camera.ui, text='', position=(0.0, -0.305),
        origin=(0.0, 0.0), scale=1.05, color=color.rgb32(235, 232, 220),
        z=-0.02,
    )
    self.message_text = Text(
        parent=camera.ui, text='', position=(0.0, -0.225),
        origin=(0.0, 0.0), scale=1.0, color=color.rgb32(220, 210, 190),
        z=-0.02,
    )
    self.message_text.enabled = False
    self.crosshair = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(0.004, 0.004), color=color.rgba32(230, 235, 245, 150),
        shader=unlit_shader, z=-0.03,
    )
    self._last_battery_pct = -1
    self.role_text.text = self._role_label()


def _game_role_label(self):
    if self.net_role == 'host':
        return ('ХОСТ • НАПАРНИК НА СВЯЗИ' if self.net_connected
                else 'ХОСТ • НАПАРНИК ОТКЛЮЧИЛСЯ')
    if self.net_role == 'client':
        return ('КЛИЕНТ • СВЯЗЬ ЕСТЬ' if self.net_connected
                else 'КЛИЕНТ • СВЯЗЬ ПОТЕРЯНА')
    return 'ОДИНОЧНАЯ ИГРА'


def _game_hud_bar(self, x, y, w, h, label, col):
    root = Entity(parent=camera.ui, position=(x, y), z=0.15)
    Entity(parent=root, model='quad', scale=(w + 0.012, h + 0.012),
           color=color.rgba32(4, 8, 14, 225), shader=unlit_shader, z=0.02)
    fill = Entity(parent=root, model='quad', origin=(-0.5, 0),
                  position=(-w / 2.0, 0), scale=(w, h), color=col,
                  shader=unlit_shader, z=0.01)
    txt = Text(parent=root, text=label, position=(-w / 2.0, h * 1.1),
               origin=(-0.5, 0), scale=0.82,
               color=color.rgba32(170, 185, 205, 255), z=0.0)
    return fill, txt


def _game_build_effects(self):
    aspect = window.aspect_ratio
    glow = self.textures.get('glow')
    vig = self.textures.get('vignette')
    face = self.textures.get('face')

    self.vignette = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(aspect, 1.0), texture=vig, color=color.white,
        shader=unlit_shader, z=0.60,
    )
    self.fear_overlay = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(aspect, 1.0), color=color.rgba32(150, 12, 12, 0),
        shader=unlit_shader, z=0.55,
    )
    self.gameover_overlay = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(aspect, 1.0), color=color.rgba32(25, 0, 0, 0),
        shader=unlit_shader, z=-0.40, enabled=False,
    )
    self.white_flash = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(aspect, 1.0), color=color.rgba32(255, 255, 255, 0),
        shader=unlit_shader, z=-0.80, enabled=False,
    )
    self.face = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(1.9, 1.6), texture=face,
        color=color.white, shader=unlit_shader, z=-0.55, enabled=False,
    )
    self.gameover_text = Text(
        parent=camera.ui, text='', position=(0.0, -0.28),
        origin=(0.0, 0.0), scale=2.1, color=color.rgb32(215, 32, 32),
        z=-0.62,
    )
    self.gameover_hint = Text(
        parent=camera.ui, text='', position=(0.0, -0.41),
        origin=(0.0, 0.0), scale=0.95, color=color.rgb32(225, 220, 215),
        z=-0.63,
    )
    self.win_overlay = Entity(
        parent=camera.ui, model='quad', position=(0, 0),
        scale=(aspect, 1.0), color=color.rgba32(0, 0, 0, 0),
        shader=unlit_shader, z=-0.50, enabled=False,
    )
    self.win_text = Text(
        parent=camera.ui, text='', position=(0.0, 0.04),
        origin=(0.0, 0.0), scale=2.3, color=color.rgb32(212, 226, 240),
        z=-0.62, enabled=False,
    )
    self.win_hint = Text(
        parent=camera.ui, text='', position=(0.0, -0.12),
        origin=(0.0, 0.0), scale=0.95, color=color.rgb32(190, 200, 215),
        z=-0.63, enabled=False,
    )


def _game_set_hud_visible(self, visible):
    for fill in (self.battery_fill, self.stamina_fill, self.fear_fill):
        root = fill.parent
        if root is not None:
            root.enabled = visible
    for element in (self.objective_text, self.keys_text,
                    self.role_text, self.crosshair):
        element.enabled = visible


def _game_show_message(self, text, duration=4.0):
    self.message_text.text = text
    self.message_text.enabled = True
    self.message_timer = duration


def _game_update_objective(self):
    if self.keys_collected >= 5:
        self.objective_text.text = 'ВСЕ КЛЮЧИ СОБРАНЫ. ВЕРНИТЕСЬ К МАШИНЕ.'
    else:
        self.objective_text.text = (f'НАЙДИТЕ КЛЮЧИ ОТ МАШИНЫ: '
                                    f'{self.keys_collected}/5')
    self.keys_text.text = f'КЛЮЧИ {self.keys_collected}/5'


def _game_update_messages(self, dt):
    if self.message_timer > 0.0:
        self.message_timer -= dt
        if self.message_timer <= 0.0:
            self.message_text.enabled = False


def _game_update_hud(self):
    bat = self.player.battery
    sta = self.player.stamina
    self.battery_fill.scale_x = max(0.001, self.battery_w * bat / 100.0)
    self.stamina_fill.scale_x = max(0.001, self.stamina_w * sta / 100.0)
    self.fear_fill.scale_x = max(0.001, self.fear_w * self.fear)
    self.battery_fill.color = (
        color.rgb32(255, 80, 60) if bat < 25
        else color.rgb32(110, 200, 255))
    self.stamina_fill.color = (
        color.rgb32(230, 120, 70) if sta < 20
        else color.rgb32(90, 220, 170))
    f = clamp01(self.fear)
    self.fear_fill.color = color.rgb32(
        255, int(220 - 180 * f), int(90 - 55 * f))
    pct = int(bat)
    if pct != self._last_battery_pct:
        self._last_battery_pct = pct
        self.battery_label.text = f'ФОНАРЬ {pct}%'
    if f > 0.45:
        pulse = 0.65 + 0.35 * pymath.sin(utime.time() * 9.0)
        a = int(95.0 * ((f - 0.45) / 0.55) * pulse)
        self.fear_overlay.color = color.rgba32(150, 12, 12, a)
    else:
        self.fear_overlay.color = color.rgba32(150, 12, 12, 0)


# ---- коллизии --------------------------------------------------------------
def _game_player_blocked(self, x, z, feet_y):
    for s in self.solids:
        if s['y0'] > feet_y + 1.3:
            continue
        if s['y1'] < feet_y + 0.25:
            continue
        if (x + PLAYER_RADIUS > s['minx'] and x - PLAYER_RADIUS < s['maxx']
                and z + PLAYER_RADIUS > s['minz']
                and z - PLAYER_RADIUS < s['maxz']):
            return True
    return False


def _game_monster_blocked(self, x, z):
    for s in self.solids:
        if s['kind'] not in ('building', 'garage', 'car'):
            continue
        if (x + 0.55 > s['minx'] and x - 0.55 < s['maxx']
                and z + 0.55 > s['minz'] and z - 0.55 < s['maxz']):
            return True
    return False


def _segment_intersects_rect(x1, z1, x2, z2, minx, maxx, minz, maxz):
    dx = x2 - x1
    dz = z2 - z1
    tmin, tmax = 0.0, 1.0
    for p, q in ((-dx, x1 - minx), (dx, maxx - x1),
                 (-dz, z1 - minz), (dz, maxz - z1)):
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                tmin = max(tmin, t)
            else:
                tmax = min(tmax, t)
            if tmin > tmax:
                return False
    return True


def _game_line_blocked(self, a, b):
    for s in self.solids:
        if s['kind'] not in ('building', 'garage'):
            continue
        if _segment_intersects_rect(a.x, a.z, b.x, b.z,
                                    s['minx'], s['maxx'],
                                    s['minz'], s['maxz']):
            return True
    return False


# ---- снег и предметы -------------------------------------------------------
def _game_update_snow(self, dt):
    p = self.player.pos
    t = utime.time()
    for f in self.snowflakes:
        e = f['e']
        pos = e.position
        pos.x += (0.7 + pymath.sin(t * 1.3 + f['phase']) * 0.5) * f['drift'] * dt
        pos.y -= f['speed'] * dt
        pos.z += (0.35 + pymath.cos(t * 1.1 + f['phase']) * 0.4) * f['drift'] * dt
        if pos.y < p.y - 2.0:
            pos.y = p.y + pyrandom.uniform(10.0, 16.0)
            pos.x = p.x + pyrandom.uniform(-22.0, 22.0)
            pos.z = p.z + pyrandom.uniform(-22.0, 22.0)
        if pos.x < p.x - 24.0:
            pos.x += 48.0
        elif pos.x > p.x + 24.0:
            pos.x -= 48.0
        if pos.z < p.z - 24.0:
            pos.z += 48.0
        elif pos.z > p.z + 24.0:
            pos.z -= 48.0
        e.position = pos


def _game_update_pickup_anim(self, dt):
    t = utime.time()
    for k in self.keys:
        if k['collected']:
            continue
        k['root'].rotation_y = (t * 55.0 + k['phase'] * 30.0) % 360.0
        k['root'].y = k['base_y'] + pymath.sin(t * 1.8 + k['phase']) * 0.08
    for b in self.batteries:
        if b['taken']:
            continue
        b['root'].rotation_y = (t * 40.0 + b['phase'] * 25.0) % 360.0
        b['root'].y = b['base_y'] + pymath.sin(t * 2.1 + b['phase']) * 0.07


# ---- восприятие и страх ----------------------------------------------------
def _game_local_target(self):
    return {
        'id': self.net_id,
        'pos': self.player.pos,
        'flashlight': bool(self.player.flashlight_on and self.player.battery > 0),
        'sprinting': self.player.sprinting,
        'moving': self.player.moving,
    }


def _game_remote_target(self):
    return {
        'id': 'client',
        'pos': self.remote['pos'],
        'flashlight': self.remote['flash'],
        'sprinting': self.remote['sprint'],
        'moving': self.remote['moving'],
    }


def _game_update_fear(self, dt):
    d = dist2d(self.player.pos, self.monster.pos)
    target = clamp01(1.0 - d / 30.0)
    if self.monster.mode == 'chase':
        target = max(target, 0.82)
    if d < 6.0:
        target = 1.0
    self.fear = lerpf(self.fear, target, min(1.0, dt * 2.0))


def _game_update_remote(self, dt):
    self.remote_player.update(dt)


def _game_update_interactions(self):
    target = self._nearest_interactable()
    if target is None:
        self.prompt_text.text = ''
        return
    kind, idx = target
    if kind == 'key':
        self.prompt_text.text = '[E] Подобрать ключ'
    elif kind == 'battery':
        self.prompt_text.text = '[E] Взять батарейку'
    else:
        if self.keys_collected >= 5:
            self.prompt_text.text = '[E] Уехать'
        else:
            self.prompt_text.text = (f'[E] Машина заперта '
                                     f'({self.keys_collected}/5)')


def _game_nearest_interactable(self):
    best = None
    best_d = 999.0
    for i, k in enumerate(self.keys):
        if k['collected']:
            continue
        d = dist2d(self.player.pos, k['pos'])
        if d < 2.7 and d < best_d:
            best = ('key', i)
            best_d = d
    for i, b in enumerate(self.batteries):
        if b['taken']:
            continue
        d = dist2d(self.player.pos, b['pos'])
        if d < 2.7 and d < best_d:
            best = ('battery', i)
            best_d = d
    d = dist2d(self.player.pos, self.car_pos)
    if d < 5.0 and d < best_d:
        best = ('car', -1)
    return best


# ---- главный цикл ----------------------------------------------------------
def _game_update(self):
    dt = min(utime.dt, 0.05)
    if self.net is not None:
        self._drain_net()
    self._update_snow(dt)
    self._update_pickup_anim(dt)
    self._update_remote(dt)

    if self.state == 'play':
        self.time_alive += dt
        self.player.update(dt)
        if self.authority:
            targets = [self._local_target()]
            if self.net_connected and self.remote['pos'] is not None:
                targets.append(self._remote_target())
            self.monster.update(dt, targets)
            self._check_catches()
        else:
            self.monster.update(dt, [])
        self._update_fear(dt)
        self._update_interactions()
        self._send_state(dt)
    elif self.state == 'dead':
        self._update_scare(dt)
    elif self.state == 'won':
        self._update_win(dt)

    self._update_messages(dt)
    self._update_hud()


def _game_input(self, key):
    if key == 'f':
        self.player.toggle_flashlight()
    elif key == 'e':
        self._interact()
    elif key == 'r' and self.state in ('dead', 'won'):
        self._request_restart()
    elif key == 'escape':
        if mouse.locked:
            mouse.locked = False
            mouse.visible = True
            self._show_message('Курсор освобождён. Esc — выйти из игры.', 3.0)
        else:
            application.quit()
    elif key == 'left mouse down' and self.state == 'play' and not mouse.locked:
        mouse.locked = True
        mouse.visible = False


# ---- сеть ------------------------------------------------------------------
def _game_send_state(self, dt):
    if not (self.net and self.net.connected):
        return
    self.net_timer += dt
    if self.net_timer < NET_RATE:
        return
    self.net_timer = 0.0
    p = self.player
    self.net.send({
        'type': 'p',
        'x': round(p.pos.x, 3), 'y': round(p.pos.y, 3),
        'z': round(p.pos.z, 3), 'ry': round(p.yaw, 2),
        'f': bool(p.flashlight_on and p.battery > 0),
        'sp': bool(p.sprinting), 'mv': bool(p.moving),
    })
    if self.authority:
        m = self.monster
        self.net.send({
            'type': 'm',
            'x': round(m.pos.x, 3), 'y': round(m.pos.y, 3),
            'z': round(m.pos.z, 3), 'ry': round(m.yaw, 2),
            'mode': m.mode,
        })


def _game_drain_net(self):
    if self.net is None:
        return
    for _ in range(80):
        try:
            msg = self.net.rx.get_nowait()
        except pyqueue.Empty:
            break
        self._handle_net(msg)


def _game_handle_net(self, msg):
    t = msg.get('type')
    if t == 'p':
        try:
            pos = Vec3(float(msg['x']), float(msg['y']), float(msg['z']))
            ry = float(msg['ry'])
        except Exception:
            return
        self.remote['pos'] = pos
        self.remote['rot'] = ry
        self.remote['flash'] = bool(msg.get('f', False))
        self.remote['sprint'] = bool(msg.get('sp', False))
        self.remote['moving'] = bool(msg.get('mv', False))
        self.remote['last'] = pytime.time()
        self.remote_player.apply_state(
            pos.x, pos.y, pos.z, ry,
            self.remote['flash'], self.remote['moving'], self.remote['sprint'])
        return

    if t == 'm':
        if not self.authority:
            try:
                self.monster.apply_network_state(
                    float(msg['x']), float(msg['y']), float(msg['z']),
                    float(msg['ry']), str(msg.get('mode', 'patrol')))
            except Exception:
                pass
        return

    if t == 'key':
        try:
            self._collect_key_local(int(msg.get('i', -1)))
        except Exception:
            pass
        return

    if t == 'battery':
        try:
            self._collect_battery_local(int(msg.get('i', -1)),
                                        by=msg.get('by'))
        except Exception:
            pass
        return

    if t == 'try_key':
        if self.authority and self.remote['pos'] is not None:
            try:
                idx = int(msg.get('i', -1))
            except Exception:
                return
            if 0 <= idx < len(self.keys) and not self.keys[idx]['collected']:
                if dist2d(self.keys[idx]['pos'], self.remote['pos']) < 3.2:
                    self._collect_key_local(idx, broadcast=self.net_connected)
        return

    if t == 'try_battery':
        if self.authority and self.remote['pos'] is not None:
            try:
                idx = int(msg.get('i', -1))
            except Exception:
                return
            if 0 <= idx < len(self.batteries) and not self.batteries[idx]['taken']:
                if dist2d(self.batteries[idx]['pos'], self.remote['pos']) < 3.2:
                    self._collect_battery_local(
                        idx, by='client', broadcast=self.net_connected)
        return

    if t == 'try_win':
        if self.authority and self.keys_collected >= 5:
            if (self.remote['pos'] is not None
                    and dist2d(self.remote['pos'], self.car_pos) < 5.5):
                self._start_win(broadcast=self.net_connected)
        return

    if t == 'game_over':
        if self.state == 'play':
            print('[СЕТЬ] Получен Game Over от хоста')
            self._trigger_game_over(
                msg.get('victim', self.net_id),
                str(msg.get('reason', '')),
                broadcast=False,
            )
        return

    if t == 'win':
        if self.state == 'play':
            print('[СЕТЬ] Получена победа от хоста')
            self._start_win(broadcast=False)
        return

    if t == 'restart_request':
        if self.authority:
            self._reset_game()
            if self.net_connected:
                self.net.send({'type': 'restart'})
        return

    if t == 'restart':
        if not self.authority:
            self._reset_game()
        return

    if t == '_disconnected':
        self._on_disconnect()


# ---- взаимодействия --------------------------------------------------------
def _game_interact(self):
    if self.state != 'play':
        return
    target = self._nearest_interactable()
    if target is None:
        return
    kind, idx = target
    if kind == 'key':
        if self.authority:
            self._collect_key_local(idx, broadcast=self.net_connected)
        elif self.net_connected:
            self.net.send({'type': 'try_key', 'i': idx})
    elif kind == 'battery':
        if self.authority:
            self._collect_battery_local(
                idx, by=self.net_id, broadcast=self.net_connected)
        elif self.net_connected:
            self.net.send({'type': 'try_battery', 'i': idx})
    elif kind == 'car':
        if self.keys_collected >= 5:
            if self.authority:
                self._start_win(broadcast=self.net_connected)
            elif self.net_connected:
                self.net.send({'type': 'try_win'})
        else:
            self._show_message(
                f'Машина заперта. Нужны все 5 ключей '
                f'({self.keys_collected}/5).', 4.0)


def _game_collect_key_local(self, idx, broadcast=False):
    if idx < 0 or idx >= len(self.keys):
        return
    k = self.keys[idx]
    if k['collected']:
        return
    k['collected'] = True
    k['root'].enabled = False
    self.keys_collected += 1
    self._update_objective()
    if self.keys_collected >= 5:
        self._show_message(
            'Все 5 ключей собраны! Вернитесь к старой машине.', 6.0)
    else:
        self._show_message(
            f'Ключ найден ({self.keys_collected}/5). '
            f'Осталось {5 - self.keys_collected}.', 4.0)
    if broadcast and self.net is not None and self.net.connected:
        self.net.send({'type': 'key', 'i': idx})


def _game_collect_battery_local(self, idx, by=None, broadcast=False):
    if idx < 0 or idx >= len(self.batteries):
        return
    b = self.batteries[idx]
    if b['taken']:
        return
    b['taken'] = True
    b['root'].enabled = False
    if by is None or by == self.net_id:
        self.player.add_battery(35.0)
        self._show_message('Батарейка найдена: +35% заряда.', 3.0)
    if broadcast and self.net is not None and self.net.connected:
        self.net.send({'type': 'battery', 'i': idx,
                       'by': by if by is not None else self.net_id})


# ---- поимка / финал --------------------------------------------------------
def _game_monster_catches(self, pos):
    d = dist2d(self.monster.pos, pos)
    dy = abs(self.monster.pos.y - pos.y)
    if d > CATCH_RADIUS or dy > 2.6:
        return False
    return not self.line_blocked(self.monster.pos, pos)


def _game_check_catches(self):
    if self.state != 'play':
        return
    if self._monster_catches(self.player.pos):
        self._trigger_game_over(
            self.net_id, 'Вас поймали', broadcast=self.net_connected)
        return
    if self.net_connected and self.remote['pos'] is not None:
        if self._monster_catches(self.remote['pos']):
            self._trigger_game_over(
                'client', 'Напарника поймали', broadcast=self.net_connected)


def _game_setup_game_over_ui(self, victim):
    self.gameover_overlay.enabled = True
    self.gameover_overlay.color = color.rgba32(30, 0, 0, 150)
    self.prompt_text.text = ''
    self.message_text.enabled = False
    self._scare_text_shown = False
    self._set_hud_visible(False)
    if victim == self.net_id:
        self.face.enabled = True
        self.face.scale = (0.35, 0.35)
        self.face.position = (0, 0, -0.55)
        self.face.rotation_z = 0
        self.white_flash.enabled = True
        self.white_flash.color = color.rgba32(255, 255, 255, 210)
        self.gameover_text.text = ''
        self.gameover_hint.text = ''
    else:
        self.face.enabled = False
        self.white_flash.enabled = False
        self.gameover_overlay.color = color.rgba32(25, 0, 0, 235)
        self.gameover_text.text = 'НАПАРНИК ПОЙМАН'
        self.gameover_hint.text = 'Полярный Скиталец забрал его. Вы остались один.'


def _game_trigger_game_over(self, victim, reason='', broadcast=True):
    if self.state != 'play':
        return
    self.state = 'dead'
    self.game_over_victim = victim
    self.scare_t = 0.0
    print(f'[ИГРА] Game Over: {victim} ({reason})')
    mouse.locked = False
    mouse.visible = True
    self.player.flashlight.enabled = False
    self._setup_game_over_ui(victim)
    if broadcast and self.net is not None and self.net.connected:
        self.net.send({'type': 'game_over', 'victim': victim,
                       'reason': reason})


def _game_update_scare(self, dt):
    self.scare_t += dt
    victim = self.game_over_victim == self.net_id
    if victim:
        t = clamp01(self.scare_t / 0.7)
        s = 0.35 + 2.1 * smoothstepf(0.0, 1.0, t)
        self.face.scale = (s * 1.25, s * 1.05)
        self.face.position = (
            pyrandom.uniform(-0.025, 0.025),
            pyrandom.uniform(-0.020, 0.020),
            -0.55,
        )
        self.face.rotation_z = pyrandom.uniform(-4.0, 4.0)
        a = 130 + int(90 * min(1.0, self.scare_t / 1.5))
        self.gameover_overlay.color = color.rgba32(35, 0, 0, a)
        if self.scare_t < 0.35:
            self.white_flash.enabled = True
            self.white_flash.color = color.rgba32(
                255, 255, 255, int(230 * (1.0 - self.scare_t / 0.35)))
        else:
            self.white_flash.enabled = False
        if self.scare_t > 1.1 and not self._scare_text_shown:
            self._scare_text_shown = True
            self.gameover_text.text = 'ИГРА ОКОНЧЕНА'
            self.gameover_hint.text = 'R — начать заново'
    else:
        a = min(235, int(235 * clamp01(self.scare_t / 0.7)))
        self.gameover_overlay.color = color.rgba32(25, 0, 0, a)
        if self.scare_t > 1.4 and not self._scare_text_shown:
            self._scare_text_shown = True
            self.gameover_hint.text = 'Полярный Скиталец забрал его. R — начать заново'


def _game_start_win(self, broadcast=True):
    if self.state != 'play':
        return
    self.state = 'won'
    self.win_t = 0.0
    mouse.locked = False
    mouse.visible = True
    self.player.flashlight.enabled = False
    self.prompt_text.text = ''
    self.message_text.enabled = False
    self._set_hud_visible(False)
    self.win_overlay.enabled = True
    self.win_overlay.color = color.rgba32(0, 0, 0, 0)
    self.win_text.text = 'ВЫ ВЫБРАЛИСЬ\n\nРоста осталась позади.'
    self.win_text.enabled = False
    self.win_hint.text = 'R — сыграть снова'
    self.win_hint.enabled = False
    if broadcast and self.net is not None and self.net.connected:
        self.net.send({'type': 'win'})


def _game_update_win(self, dt):
    self.win_t += dt
    a = int(240 * clamp01(self.win_t / 1.6))
    self.win_overlay.color = color.rgba32(0, 0, 0, a)
    if self.win_t > 0.8:
        self.win_text.enabled = True
    if self.win_t > 1.6:
        self.win_hint.enabled = True


# ---- рестарт ---------------------------------------------------------------
def _game_request_restart(self):
    if self.authority:
        self._reset_game()
        if self.net_connected:
            self.net.send({'type': 'restart'})
    elif self.net_connected:
        self.net.send({'type': 'restart_request'})
        self._show_message('Запрос на рестарт отправлен хосту...', 3.0)
    else:
        self._reset_game()


def _game_reset_game(self):
    self.state = 'play'
    self.time_alive = 0.0
    self.fear = 0.0
    self.keys_collected = 0
    self.message_timer = 0.0
    self.net_timer = 0.0
    self.monster_net_timer = 0.0
    self.scare_t = 0.0
    self.win_t = 0.0
    self.game_over_victim = None
    self._scare_text_shown = False

    self.player.reset()
    self.monster.reset()
    self.monster.network_controlled = (not self.authority) and self.net_connected
    for k in self.keys:
        k['collected'] = False
        k['root'].enabled = True
    for b in self.batteries:
        b['taken'] = False
        b['root'].enabled = True
    self.remote_player.reset()
    self.remote['pos'] = None
    self.remote['last'] = 0.0

    self.gameover_overlay.enabled = False
    self.face.enabled = False
    self.white_flash.enabled = False
    self.gameover_text.text = ''
    self.gameover_hint.text = ''
    self.win_overlay.enabled = False
    self.win_text.enabled = False
    self.win_hint.enabled = False
    self.fear_overlay.color = color.rgba32(150, 12, 12, 0)
    self.prompt_text.text = ''
    self.message_text.enabled = False
    self._set_hud_visible(True)

    self._update_objective()
    self._last_battery_pct = -1
    self.role_text.text = self._role_label()
    self._show_message('Новая попытка. Найдите 5 ключей от машины.', 5.0)
    mouse.locked = True
    mouse.visible = False

    if self.net is not None:
        while not self.net.rx.empty():
            try:
                self.net.rx.get_nowait()
            except pyqueue.Empty:
                break


def _game_on_disconnect(self):
    if not self.net_connected:
        return
    self.net_connected = False
    self.remote_player.reset()
    self.remote['pos'] = None
    self.role_text.text = self._role_label()
    self._show_message(
        'Связь потеряна. Игра продолжается в одиночном режиме.', 7.0)
    if self.net_role == 'client':
        self.authority = True
        self.monster.network_controlled = False
        self.monster.mode = 'patrol'
        self.monster.waypoint_index = 1


# ---- автотест --------------------------------------------------------------
def _game_shot(self, name):
    path = pyos.path.join(
        tempfile.gettempdir(), f'skalnaya_{self.net_role}_{name}.png')
    try:
        base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print(f'[SMOKE] screenshot: {path}')
    except Exception as exc:
        print(f'[SMOKE] screenshot failed: {exc}')


def _game_smoke_normal(self):
    self._shot('normal')


def _game_smoke_scare(self):
    if self.state == 'play' and self.authority:
        print('[SMOKE] triggering game over')
        self._trigger_game_over(self.net_id, 'smoke-test',
                                broadcast=self.net_connected)


def _game_smoke_shot(self):
    self._shot('gameover')


# ---- привязка методов к Game ----------------------------------------------
Game._build_ui = _game_build_ui
Game._role_label = _game_role_label
Game._hud_bar = _game_hud_bar
Game._build_effects = _game_build_effects
Game._show_message = _game_show_message
Game._set_hud_visible = _game_set_hud_visible
Game._update_objective = _game_update_objective
Game._update_messages = _game_update_messages
Game._update_hud = _game_update_hud
Game.player_blocked = _game_player_blocked
Game.monster_blocked = _game_monster_blocked
Game.line_blocked = _game_line_blocked
Game._update_snow = _game_update_snow
Game._update_pickup_anim = _game_update_pickup_anim
Game._local_target = _game_local_target
Game._remote_target = _game_remote_target
Game._update_fear = _game_update_fear
Game._update_remote = _game_update_remote
Game._update_interactions = _game_update_interactions
Game._nearest_interactable = _game_nearest_interactable
Game.update = _game_update
Game.input = _game_input
Game._send_state = _game_send_state
Game._drain_net = _game_drain_net
Game._handle_net = _game_handle_net
Game._interact = _game_interact
Game._collect_key_local = _game_collect_key_local
Game._collect_battery_local = _game_collect_battery_local
Game._monster_catches = _game_monster_catches
Game._check_catches = _game_check_catches
Game._setup_game_over_ui = _game_setup_game_over_ui
Game._trigger_game_over = _game_trigger_game_over
Game._update_scare = _game_update_scare
Game._start_win = _game_start_win
Game._update_win = _game_update_win
Game._request_restart = _game_request_restart
Game._reset_game = _game_reset_game
Game._on_disconnect = _game_on_disconnect
Game._shot = _game_shot
Game._smoke_normal = _game_smoke_normal
Game._smoke_scare = _game_smoke_scare
Game._smoke_shot = _game_smoke_shot


# ----------------------------------------------------------------------------
# ЗАПУСК
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Роста — полярная ночь (Ursina, кооператив)')
    p.add_argument('--host', action='store_true', help='запустить сервер')
    p.add_argument('--client', type=str, default=None, metavar='IP',
                   help='подключиться к хосту')
    p.add_argument('--single', action='store_true', help='одиночная игра')
    p.add_argument('--port', type=int, default=DEFAULT_PORT,
                   help=f'TCP-порт (по умолчанию {DEFAULT_PORT})')
    p.add_argument('--smoke-test', action='store_true',
                   help='автотест: скриншоты и выход')
    return p.parse_args()


def choose_mode(args):
    if args.host:
        return 'host', None
    if args.client:
        return 'client', args.client
    if args.single or args.smoke_test:
        return 'solo', None

    print('=' * 58)
    print('  Роста — ПОЛЯРНАЯ НОЧЬ')
    print('  3D-хоррор в Мурманске • кооператив на 2 игрока не работает если че')
    print('=' * 58)
    print('  1 — Сервер (хост)')
    print('  2 — Клиент (подключиться по Radmin VPN)')
    print('  3 — Одиночная игра')
    print('=' * 58)
    while True:
        try:
            c = builtins.input('Выбор [1/2/3]: ').strip()
        except (EOFError, KeyboardInterrupt):
            return 'solo', None
        if c == '1':
            return 'host', None
        if c == '2':
            try:
                ip = builtins.input(
                    'IP хоста (Radmin VPN, например 26.12.34.56): ').strip()
            except (EOFError, KeyboardInterrupt):
                return 'solo', None
            return 'client', ip
        if c == '3':
            return 'solo', None
        print('Введите 1, 2 или 3.')


GAME = None


def update():
    if GAME is not None:
        GAME.update()


def input(key):
    if GAME is not None:
        GAME.input(key)


def main():
    global GAME
    args = parse_args()
    role, ip = choose_mode(args)

    net = None
    if role == 'host':
        net = Network()
        if not net.host(args.port, timeout=300.0):
            print('Не удалось запустить сервер. Порт занят?')
            return
    elif role == 'client':
        if not ip:
            print('Не указан IP хоста.')
            return
        port = args.port
        if ':' in ip:
            ip, port_s = ip.rsplit(':', 1)
            try:
                port = int(port_s)
            except ValueError:
                pass
        net = Network()
        if not net.connect(ip, port, timeout=25.0):
            print(f'Не удалось подключиться к {ip}:{port}.')
            print(f'Причина: {net.last_error or "неизвестная ошибка"}')
            print('Проверьте, что хост уже запустил игру и ожидает клиента,')
            print('оба компьютера подключены к одной сети Radmin VPN,')
            print('а введён именно Radmin IP хоста (обычно начинается с 26.).')
            print('На компьютере хоста разрешите Python/Ursina в брандмауэре')
            print(f'для входящих подключений TCP-порта {port}.')
            try:
                builtins.input('Нажмите Enter, чтобы закрыть это окно...')
            except (EOFError, KeyboardInterrupt):
                pass
            return

    app = Ursina()
    GAME = Game(net=net, net_role=role, smoke_test=args.smoke_test)
    app.run()
    if net is not None:
        net.close()


if __name__ == '__main__':
    main()
