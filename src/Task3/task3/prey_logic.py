"""Escape planner for the prey (pure Python, no ROS).

The prey samples candidate moves (x, y, theta) on a circle around itself and
scores each one:

* distance from where the predator is expected to be after the move
  (the predator is assumed to step straight towards the candidate point),
* a penalty for ending close to an arena edge, growing quadratically so the
  prey turns *before* it reaches the wall,
* a hard veto for leaving the arena or touching an (inflated) obstacle,
* a soft penalty for passing near an obstacle,
* a penalty for changing the previously chosen heading (commitment), which
  stops the prey dithering between two equally good escape routes,
* a look-ahead wall check twice as far out, so it starts curving away from
  an edge early instead of running into a corner.

The best-scoring move is returned and executed by the node's controller.
"""

import math

from .common import ARENA_MAX, ARENA_MIN, distance, segment_clearance, wrap_angle

VETO = -1e6


def _wall_distance(x, y):
    return min(x - ARENA_MIN, ARENA_MAX - x, y - ARENA_MIN, ARENA_MAX - y)


def score_move(px, py, ptheta, heading, threats, obstacles, step,
               threat_step, wall_margin, obstacle_margin, previous=None):
    cx = px + step * math.cos(heading)
    cy = py + step * math.sin(heading)

    # Hard constraints: stay inside the arena and out of obstacles.
    if not (ARENA_MIN + 0.3 < cx < ARENA_MAX - 0.3
            and ARENA_MIN + 0.3 < cy < ARENA_MAX - 0.3):
        return VETO, cx, cy
    soft = 0.0
    for ox, oy, r in obstacles:
        clearance = segment_clearance(px, py, cx, cy, ox, oy) - r
        current = distance(px, py, ox, oy) - r
        # Veto moves that get too close, but always allow moving away.
        if clearance < 0.5 and clearance < current - 1e-3:
            return VETO, cx, cy
        if clearance < obstacle_margin:
            soft -= 2.5 * (obstacle_margin - clearance)

    # Distance to each threat after it has moved towards the candidate point.
    score = 0.0
    nearest = float('inf')
    for tx, ty in threats:
        d_now = distance(tx, ty, cx, cy)
        frac = min(1.0, threat_step / d_now) if d_now > 1e-6 else 1.0
        fx, fy = tx + frac * (cx - tx), ty + frac * (cy - ty)
        d_after = distance(fx, fy, cx, cy)
        nearest = min(nearest, d_after)
        score -= 4.0 / (d_after + 0.3)  # every threat pushes, close ones more
    score += 1.5 * nearest

    # Wall penalty (quadratic inside the margin) now and further ahead.
    wall = _wall_distance(cx, cy)
    if wall < wall_margin:
        score -= 6.0 * (wall_margin - wall) ** 2
    ahead = _wall_distance(px + 2.0 * step * math.cos(heading),
                           py + 2.0 * step * math.sin(heading))
    if ahead < wall_margin:
        score -= 2.0 * (wall_margin - ahead) ** 2

    # Smoothness and commitment to the previous decision.
    score -= 0.4 * abs(wrap_angle(heading - ptheta)) / math.pi
    if previous is not None:
        score -= 1.5 * abs(wrap_angle(heading - previous)) / math.pi
    return score + soft, cx, cy


def plan_escape(px, py, ptheta, threats, obstacles=(), step=1.2,
                threat_step=1.2, n_candidates=32, wall_margin=1.8,
                obstacle_margin=1.0, previous=None):
    """Return the best move (x, y, theta) and its score.

    ``threats`` is a list of (x, y) predator positions; ``previous`` is the
    heading chosen at the last planning step (or None).
    """
    best = (VETO, px, py, ptheta)
    for k in range(n_candidates):
        heading = wrap_angle(2.0 * math.pi * k / n_candidates)
        score, cx, cy = score_move(px, py, ptheta, heading, threats,
                                   obstacles, step, threat_step, wall_margin,
                                   obstacle_margin, previous)
        if score > best[0]:
            best = (score, cx, cy, heading)
    score, x, y, theta = best
    if score <= VETO:
        # Boxed in: head for the middle of the arena.
        mid = (ARENA_MIN + ARENA_MAX) / 2.0
        theta = math.atan2(mid - py, mid - px)
        x, y = px + step * math.cos(theta), py + step * math.sin(theta)
    return (x, y, theta), score
