"""ROS-free core of Task 4: predator policy, learning prey and a fast simulator.

The same functions drive the ROS nodes (``predator.py`` / ``prey.py``) and the
offline trainer (``train_prey.py``), so what the prey learns offline is
exactly what it does in turtlesim.

Learning prey: tabular Q-learning
---------------------------------
State (discretised, expressed relative to the nearest predator so the same
situation in any orientation maps to the same state):

* distance to the nearest predator: 4 bins,
* bearing of the second-nearest predator relative to the nearest one:
  8 sectors, or "far" if it is more than 4 m away,
* bearing of the nearest hazard (wall or obstacle) relative to the nearest
  predator: 8 sectors, or "none" if nothing is within 1.5 m.

Actions: 8 escape headings, expressed as offsets from "directly away from
the nearest predator" (0, +-45, +-90, +-135, 180 deg).  Action 0 is listed
first so ties (e.g. an untrained table) default to the classic
"run directly away" behaviour; learning then refines it.

Safety shield: actions whose next 1 m would leave the arena or hit an
obstacle are masked out before the agent chooses, so exploration can never
drive the robot into an obstacle.

Reward: +0.1 per decision survived, plus 0.5 x the change in distance to
the nearest predator, -10 when captured, +10 for surviving the time limit.
"""

import json
import math
import os
import random

from .common import (ARENA_MAX, ARENA_MIN, avoidance_heading, clamp,
                     cruise_controller, distance, heading_controller,
                     random_free_point, segment_clearance, wrap_angle)
from .obstacles import OBSTACLES

PREDATOR_NAMES = ['predator1', 'predator2', 'predator3']
PREDATOR_SPEED = 1.5
PREY_SPEED = 1.9
CAPTURE_DISTANCE = 0.5
TIME_LIMIT = 60.0
DECISION_PERIOD = 0.3
# Lead factor per predator: 0 = chase where the prey is, 1 = full intercept.
PREDATOR_LEAD = {'predator1': 0.0, 'predator2': 0.6, 'predator3': 1.0}

ACTION_OFFSETS = [0, 1, -1, 2, -2, 3, -3, 4]   # x 45 degrees from "away"
N_ACTIONS = len(ACTION_OFFSETS)
_DIST_BINS = (1.5, 3.0, 5.0)
_SECOND_FAR = 4.0
_HAZARD_RANGE = 1.5
_SHIELD_STEP = 1.0


# --------------------------------------------------------------------------
# Predators
# --------------------------------------------------------------------------
def predator_command(px, py, ptheta, prey_x, prey_y, prey_vx, prey_vy,
                     teammates=(), lead=0.0, max_speed=PREDATOR_SPEED,
                     obstacles=OBSTACLES):
    """(linear, angular) command for one independent predator.

    It aims at the prey's position extrapolated ``lead`` x time-to-reach
    ahead (constant-velocity intercept) and steers round obstacles.  Other
    predators are treated as small obstacles purely for collision safety.
    """
    reach = distance(px, py, prey_x, prey_y) / max_speed
    t = lead * min(reach, 2.5)
    tx = clamp(prey_x + prey_vx * t, ARENA_MIN + 0.3, ARENA_MAX - 0.3)
    ty = clamp(prey_y + prey_vy * t, ARENA_MIN + 0.3, ARENA_MAX - 0.3)
    near_mates = [(mx, my, 0.15) for mx, my in teammates
                  if distance(px, py, mx, my) < 1.5]
    heading = avoidance_heading(px, py, tx, ty, list(obstacles) + near_mates)
    return heading_controller(ptheta, heading, max_speed)


# --------------------------------------------------------------------------
# Learning prey
# --------------------------------------------------------------------------
def _sector(angle):
    """Map an angle (rad, relative) to one of 8 sectors centred on 0."""
    return int(round(wrap_angle(angle) / (math.pi / 4))) % 8


def ray_length(x, y, angle, obstacles=OBSTACLES, max_range=5.0):
    c, s = math.cos(angle), math.sin(angle)
    best = max_range
    if c > 1e-9:
        best = min(best, (ARENA_MAX - x) / c)
    elif c < -1e-9:
        best = min(best, (ARENA_MIN - x) / c)
    if s > 1e-9:
        best = min(best, (ARENA_MAX - y) / s)
    elif s < -1e-9:
        best = min(best, (ARENA_MIN - y) / s)
    for ox, oy, r in obstacles:
        fx, fy = x - ox, y - oy
        b = fx * c + fy * s
        disc = b * b - (fx * fx + fy * fy - r * r)
        if disc >= 0.0:
            t = -b - math.sqrt(disc)
            if 0.0 <= t < best:
                best = t
    return max(best, 0.0)


class PreyAgent:
    """Tabular Q-learning agent with a safety shield."""

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2,
                 epsilon_min=0.02, epsilon_decay=0.9995, alpha_min=0.01,
                 alpha_decay=0.9999, seed=None):
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = 0
        self.q = {}
        self.rng = random.Random(seed)

    # --------------------------------------------------------------- state
    @staticmethod
    def away_angle(prey_xy, predators):
        """Direction pointing directly away from the nearest predator."""
        qx, qy = prey_xy
        nx, ny = min(predators, key=lambda p: distance(qx, qy, *p))
        return math.atan2(qy - ny, qx - nx)

    def encode(self, prey_xy, predators, obstacles=OBSTACLES):
        qx, qy = prey_xy
        ranked = sorted(predators, key=lambda p: distance(qx, qy, *p))
        nx, ny = ranked[0]
        d1 = distance(qx, qy, nx, ny)
        toward = math.atan2(ny - qy, nx - qx)
        d_bin = sum(d1 > b for b in _DIST_BINS)

        second = 8
        if len(ranked) > 1:
            sx, sy = ranked[1]
            if distance(qx, qy, sx, sy) < _SECOND_FAR:
                second = _sector(math.atan2(sy - qy, sx - qx) - toward)

        hazard = 8
        best = _HAZARD_RANGE
        for k in range(8):
            angle = toward + k * math.pi / 4
            r = ray_length(qx, qy, angle, obstacles, _HAZARD_RANGE + 0.1)
            if r < best:
                best, hazard = r, k
        return f'{d_bin}|{second}|{hazard}'

    def action_heading(self, action, away):
        return wrap_angle(away + ACTION_OFFSETS[action] * math.pi / 4)

    def legal_actions(self, prey_xy, away, obstacles=OBSTACLES):
        """Safety shield: actions whose next metre stays in free space."""
        qx, qy = prey_xy
        legal = []
        for a in range(N_ACTIONS):
            h = self.action_heading(a, away)
            cx = qx + _SHIELD_STEP * math.cos(h)
            cy = qy + _SHIELD_STEP * math.sin(h)
            if not (ARENA_MIN + 0.4 < cx < ARENA_MAX - 0.4
                    and ARENA_MIN + 0.4 < cy < ARENA_MAX - 0.4):
                continue
            current = [distance(qx, qy, ox, oy) - r
                       for ox, oy, r in obstacles]
            blocked = False
            for (ox, oy, r), cur in zip(obstacles, current):
                clear = segment_clearance(qx, qy, cx, cy, ox, oy) - r
                if clear < 0.45 and clear < cur - 1e-3:
                    blocked = True
                    break
            if not blocked:
                legal.append(a)
        return legal or list(range(N_ACTIONS))

    # ------------------------------------------------------------ learning
    def values(self, state):
        if state not in self.q:
            self.q[state] = [0.0] * N_ACTIONS
        return self.q[state]

    def choose(self, state, legal, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        values = self.values(state)
        # max() keeps the first of equal values -> prefers "away" on ties.
        return max(legal, key=lambda a: (values[a], -a))

    def update(self, state, action, reward, next_state, next_legal,
               terminal):
        values = self.values(state)
        target = reward
        if not terminal:
            nxt = self.values(next_state)
            target += self.gamma * max(nxt[a] for a in next_legal)
        values[action] += self.alpha * (target - values[action])

    def end_episode(self):
        self.episodes += 1
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    # ------------------------------------------------------------- storage
    def save(self, path):
        data = {'episodes': self.episodes, 'epsilon': self.epsilon,
                'alpha': self.alpha, 'gamma': self.gamma,
                'q': {k: [round(v, 4) for v in vals]
                      for k, vals in self.q.items()}}
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.episodes = data.get('episodes', 0)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.alpha = data.get('alpha', self.alpha)
        self.q = {k: list(v) for k, v in data.get('q', {}).items()}
        return True


def step_reward(old_min, new_min):
    return 0.1 + 0.5 * clamp(new_min - old_min, -1.0, 1.0)


CAPTURE_REWARD = -10.0
SURVIVE_REWARD = 10.0


# --------------------------------------------------------------------------
# Offline simulator (turtlesim-equivalent unicycle kinematics)
# --------------------------------------------------------------------------
class Turtle:
    __slots__ = ('x', 'y', 'theta', 'v')

    def __init__(self, x, y, theta):
        self.x, self.y, self.theta, self.v = x, y, theta, 0.0

    def step(self, linear, angular, dt):
        self.theta = wrap_angle(self.theta + angular * dt)
        self.x = clamp(self.x + linear * math.cos(self.theta) * dt,
                       ARENA_MIN, ARENA_MAX)
        self.y = clamp(self.y + linear * math.sin(self.theta) * dt,
                       ARENA_MIN, ARENA_MAX)
        self.v = linear


def random_world(rng, n_predators=3):
    prey_xy = random_free_point(rng, OBSTACLES)
    taken = [prey_xy]
    preds = []
    for _ in range(n_predators):
        p = random_free_point(rng, OBSTACLES, others=taken,
                              min_separation=3.0)
        taken.append(p)
        preds.append(p)
    prey = Turtle(*prey_xy, rng.uniform(-math.pi, math.pi))
    predators = [Turtle(x, y, rng.uniform(-math.pi, math.pi))
                 for x, y in preds]
    return prey, predators


def run_episode(agent, rng, policy='q', learn=True, dt=0.1):
    """Simulate one game. policy: 'q', 'away' (flee nearest) or 'random'.

    Returns (survived: bool, time_s: float).
    """
    prey, predators = random_world(rng)
    steps_per_decision = max(1, round(DECISION_PERIOD / dt))
    state = action = None
    heading = prey.theta
    old_min = None
    t = 0.0
    step = 0
    while True:
        pred_xy = [(p.x, p.y) for p in predators]
        d_min = min(distance(prey.x, prey.y, x, y) for x, y in pred_xy)
        captured = d_min < CAPTURE_DISTANCE
        timeout = t >= TIME_LIMIT
        if captured or timeout:
            if learn and state is not None:
                reward = CAPTURE_REWARD if captured else SURVIVE_REWARD
                agent.update(state, action, reward, None, None, True)
            if learn:
                agent.end_episode()
            return (not captured), t

        if step % steps_per_decision == 0:
            away = agent.away_angle((prey.x, prey.y), pred_xy)
            new_state = agent.encode((prey.x, prey.y), pred_xy)
            legal = agent.legal_actions((prey.x, prey.y), away)
            if learn and state is not None:
                agent.update(state, action, step_reward(old_min, d_min),
                             new_state, legal, False)
            if policy == 'q':
                action = agent.choose(new_state, legal, explore=learn)
            elif policy == 'away':
                action = legal[0]
            else:
                action = rng.choice(legal)
            state, old_min = new_state, d_min
            heading = agent.action_heading(action, away)

        lin, ang = cruise_controller(prey.theta, heading, PREY_SPEED)
        vx = prey.v * math.cos(prey.theta)
        vy = prey.v * math.sin(prey.theta)
        commands = []
        for i, (p, name) in enumerate(zip(predators, PREDATOR_NAMES)):
            mates = [xy for j, xy in enumerate(pred_xy) if j != i]
            commands.append(predator_command(
                p.x, p.y, p.theta, prey.x, prey.y, vx, vy, mates,
                PREDATOR_LEAD[name]))
        prey.step(lin, ang, dt)
        for p, (l, a) in zip(predators, commands):
            p.step(l, a, dt)
        t += dt
        step += 1
