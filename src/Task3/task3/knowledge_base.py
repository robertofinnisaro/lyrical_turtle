"""Predator knowledge base: storage in ``predator-kb.xml`` plus prediction.

File format (as required by the brief)::

    <xml>
      <trial>
        <number>1</number>
        <move number="1">
          <predator>3,3</predator>
          <prey>10,10</prey>
        </move>
        ...
        <result>capture</result>
      </trial>
    </xml>

Learning method
---------------
Every recorded turn ``j`` of every previous trial is a training example.
Its *situation* is described in the prey's own frame of reference (x axis =
the direction the prey moved during the last turn), so a manoeuvre learnt
running clockwise along one wall also applies running anticlockwise along
another:

* where the predator is relative to the prey (ahead/behind, left/right),
* how fast the prey was moving,
* how much free space the prey has ahead, to its left and to its right
  (ray casts against the arena walls and obstacles).

Its *outcome* is where the prey actually was ``H`` turns later, also in the
prey frame.  At run time the predator builds the same description of the
current situation, finds the k most similar past situations
(k-nearest-neighbour regression), averages their outcomes, rotates the
result back into the world frame and adds it to the prey's position.  That
is the *virtual target*: the likely future prey position.  ``H`` is chosen
so the predator needs about ``H`` turns to reach the virtual target (an
intercept), and the prediction is shrunk back towards the live position
when no past situation is very similar (low confidence).

The method is pure Python, ROS-free, and is unit-testable offline.
"""

import math
import os
import xml.etree.ElementTree as ET

from .common import ARENA_MAX, ARENA_MIN

# Relative importance of the feature components.
_W_OFFSET = 0.6
_W_SPEED = 2.0
_W_SPACE = 0.8
_RAY_MAX = 5.0
_MIN_MOTION = 0.05  # m per turn below which the prey counts as stationary


class KnowledgeBase:

    def __init__(self, path, obstacles=(), k=6, max_horizon=8,
                 confidence_scale=1.5):
        self.path = path
        self.obstacles = list(obstacles)
        self.k = k
        self.max_horizon = max_horizon
        self.confidence_scale = confidence_scale
        self.trials = []   # list of dicts: number, moves[(px,py,qx,qy)], result
        self._samples = []  # (feature, heading, trial index, move index)

    # ------------------------------------------------------------------ I/O
    def load(self):
        """Load previous trials if the XML file exists. Returns #trials."""
        self.trials = []
        if os.path.exists(self.path):
            root = ET.parse(self.path).getroot()
            for trial in root.findall('trial'):
                moves = []
                for move in sorted(trial.findall('move'),
                                   key=lambda m: int(m.get('number', 0))):
                    px, py = _parse_xy(move.findtext('predator'))
                    qx, qy = _parse_xy(move.findtext('prey'))
                    moves.append((px, py, qx, qy))
                self.trials.append({
                    'number': int(trial.findtext('number', '0')),
                    'moves': moves,
                    'result': trial.findtext('result', 'unknown'),
                })
        self._build_index()
        return len(self.trials)

    def append_trial(self, moves, result):
        """Add a finished trial and rewrite the XML file. Returns its number."""
        number = max((t['number'] for t in self.trials), default=0) + 1
        self.trials.append({'number': number, 'moves': list(moves),
                            'result': result})
        root = ET.Element('xml')
        for trial in self.trials:
            t_el = ET.SubElement(root, 'trial')
            ET.SubElement(t_el, 'number').text = str(trial['number'])
            for i, (px, py, qx, qy) in enumerate(trial['moves'], start=1):
                m_el = ET.SubElement(t_el, 'move', number=str(i))
                ET.SubElement(m_el, 'predator').text = f'{px:.2f},{py:.2f}'
                ET.SubElement(m_el, 'prey').text = f'{qx:.2f},{qy:.2f}'
            ET.SubElement(t_el, 'result').text = trial['result']
        tree = ET.ElementTree(root)
        if hasattr(ET, 'indent'):
            ET.indent(tree, space='  ')
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tree.write(self.path, encoding='unicode')
        self._build_index()
        return number

    # ------------------------------------------------------------ learning
    @property
    def num_samples(self):
        return len(self._samples)

    def _build_index(self):
        self._samples = []
        for ti, trial in enumerate(self.trials):
            moves = trial['moves']
            for j in range(1, len(moves) - 1):
                px, py, qx, qy = moves[j]
                _, _, qx0, qy0 = moves[j - 1]
                feature, heading = self._feature(px, py, qx, qy,
                                                 qx - qx0, qy - qy0)
                self._samples.append((feature, heading, ti, j))

    def predict(self, pred_x, pred_y, prey_x, prey_y, prey_dx, prey_dy,
                predator_step):
        """Predict where the prey will be when the predator can get there.

        ``prey_dx, prey_dy``: prey displacement over the last turn.
        ``predator_step``: metres the predator covers per turn.
        Returns (x, y, confidence, horizon_turns).
        """
        if not self._samples:
            return prey_x, prey_y, 0.0, 0
        query, heading = self._feature(pred_x, pred_y, prey_x, prey_y,
                                       prey_dx, prey_dy)
        scored = sorted(
            (_sq_dist(query, f), h, ti, j) for f, h, ti, j in self._samples)
        neighbours = scored[:self.k]
        nearest = math.sqrt(neighbours[0][0])
        confidence = math.exp(-nearest / self.confidence_scale)
        cos_q, sin_q = math.cos(heading), math.sin(heading)

        tx, ty = prey_x, prey_y
        horizon = 1
        for _ in range(3):  # fixed-point iteration for the intercept horizon
            reach = math.hypot(tx - pred_x, ty - pred_y)
            horizon = max(1, min(self.max_horizon,
                                 round(reach / max(predator_step, 1e-3))))
            fwd = left = wsum = 0.0
            for d2, h, ti, j in neighbours:
                moves = self.trials[ti]['moves']
                end = min(j + horizon, len(moves) - 1)
                dx = moves[end][2] - moves[j][2]
                dy = moves[end][3] - moves[j][3]
                # Outcome expressed in that sample's prey frame.
                c, s = math.cos(h), math.sin(h)
                w = 1.0 / (math.sqrt(d2) + 0.1)
                fwd += w * (c * dx + s * dy)
                left += w * (-s * dx + c * dy)
                wsum += w
            fwd, left = confidence * fwd / wsum, confidence * left / wsum
            tx = prey_x + cos_q * fwd - sin_q * left
            ty = prey_y + sin_q * fwd + cos_q * left
            tx = min(max(tx, ARENA_MIN + 0.2), ARENA_MAX - 0.2)
            ty = min(max(ty, ARENA_MIN + 0.2), ARENA_MAX - 0.2)
        return tx, ty, confidence, horizon

    # ------------------------------------------------------------ features
    def _feature(self, px, py, qx, qy, vx, vy):
        """Situation descriptor in the prey frame; returns (feature, heading).

        If the prey hardly moved, its heading is taken as "away from the
        predator", which is how the prey behaves when it starts running.
        """
        speed = math.hypot(vx, vy)
        if speed > _MIN_MOTION:
            heading = math.atan2(vy, vx)
        else:
            heading = math.atan2(qy - py, qx - px)
        c, s = math.cos(heading), math.sin(heading)
        ox, oy = px - qx, py - qy  # predator relative to prey
        rel_fwd = c * ox + s * oy
        rel_left = -s * ox + c * oy
        ahead = self._ray(qx, qy, heading)
        left = self._ray(qx, qy, heading + math.pi / 2)
        right = self._ray(qx, qy, heading - math.pi / 2)
        feature = (rel_fwd * _W_OFFSET, rel_left * _W_OFFSET,
                   speed * _W_SPEED,
                   ahead * _W_SPACE, left * _W_SPACE, right * _W_SPACE)
        return feature, heading

    def _ray(self, x, y, angle):
        """Free distance from (x, y) along ``angle`` (walls and obstacles)."""
        c, s = math.cos(angle), math.sin(angle)
        best = _RAY_MAX
        if c > 1e-9:
            best = min(best, (ARENA_MAX - x) / c)
        elif c < -1e-9:
            best = min(best, (ARENA_MIN - x) / c)
        if s > 1e-9:
            best = min(best, (ARENA_MAX - y) / s)
        elif s < -1e-9:
            best = min(best, (ARENA_MIN - y) / s)
        for ox, oy, r in self.obstacles:
            # Ray / circle intersection.
            fx, fy = x - ox, y - oy
            b = fx * c + fy * s
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc >= 0.0:
                t = -b - math.sqrt(disc)
                if 0.0 <= t < best:
                    best = t
        return max(best, 0.0)


def _parse_xy(text):
    x, y = (float(v) for v in (text or '0,0').split(','))
    return x, y


def _sq_dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))
