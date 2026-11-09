"""Helpers shared by every node in this package.

This file is deliberately ROS-light: the geometry / navigation helpers are
plain Python so they can be unit tested (and reused by offline simulators)
without a running ROS graph.  Only ``WorldBuilder`` touches rclpy.
"""

import math
import random
import time

# turtlesim's interfaces moved to the ``turtlesim_msgs`` package in Jazzy
# (and so Kilted / Lyrical).  Humble still ships them inside ``turtlesim``.
try:
    from turtlesim_msgs.msg import Pose
    from turtlesim_msgs.srv import Kill, SetPen, Spawn, TeleportAbsolute
except ImportError:  # pragma: no cover - depends on the ROS distro
    try:
        from turtlesim.msg import Pose
        from turtlesim.srv import Kill, SetPen, Spawn, TeleportAbsolute
    except ImportError:  # no ROS at all: offline simulation / unit tests
        Pose = Kill = SetPen = Spawn = TeleportAbsolute = None

__all__ = [
    'Pose', 'Kill', 'SetPen', 'Spawn', 'TeleportAbsolute',
    'ARENA_MIN', 'ARENA_MAX', 'wrap_angle', 'distance', 'clamp',
    'random_free_point', 'point_in_obstacle', 'segment_clearance',
    'avoidance_heading', 'heading_controller', 'latched_qos', 'WorldBuilder',
    'make_rng', 'cruise_controller',
]

# turtlesim's window is 500 px wide at 45 px per metre -> 11.0889 m.
ARENA_MIN = 0.0
ARENA_MAX = 11.088889


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def wrap_angle(angle):
    """Wrap an angle to the interval (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def distance(ax, ay, bx, by):
    """Euclidean distance between two points."""
    return math.hypot(bx - ax, by - ay)


def clamp(value, low, high):
    return max(low, min(high, value))


def point_in_obstacle(x, y, obstacles, inflate=0.0):
    """True if (x, y) lies inside any (inflated) circular obstacle."""
    for ox, oy, r in obstacles:
        if distance(x, y, ox, oy) < r + inflate:
            return True
    return False


def segment_clearance(ax, ay, bx, by, cx, cy):
    """Shortest distance from point C to the segment AB."""
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq < 1e-12:
        return distance(ax, ay, cx, cy)
    t = clamp(((cx - ax) * abx + (cy - ay) * aby) / length_sq, 0.0, 1.0)
    return distance(ax + t * abx, ay + t * aby, cx, cy)


def random_free_point(rng, obstacles=(), others=(), wall_margin=1.0,
                      obstacle_margin=0.8, min_separation=3.0, attempts=500):
    """Pick a random spawn point away from walls, obstacles and other robots."""
    lo, hi = ARENA_MIN + wall_margin, ARENA_MAX - wall_margin
    for _ in range(attempts):
        x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if point_in_obstacle(x, y, obstacles, obstacle_margin):
            continue
        if any(distance(x, y, ox, oy) < min_separation for ox, oy in others):
            continue
        return x, y
    # Fall back to a less strict separation rather than failing outright.
    return random_free_point(rng, obstacles, others, wall_margin,
                             obstacle_margin, min_separation * 0.7, attempts)


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
def avoidance_heading(x, y, gx, gy, obstacles=(), influence=1.2, inflate=0.45):
    """Heading (rad) towards goal (gx, gy) that bends around circular obstacles.

    A goal-attraction unit vector is combined with, for every obstacle that is
    both close and in the way, a tangential "slide around" vector (on the side
    the goal lies) plus a radial push-away that grows as clearance shrinks.
    This is a simple potential-field / vector-field-histogram hybrid; the
    tangential term stops the robot getting stuck in the classic local
    minimum directly in front of a round obstacle.
    """
    ax, ay = gx - x, gy - y
    goal_dist = math.hypot(ax, ay)
    if goal_dist < 1e-6:
        return 0.0
    ax, ay = ax / goal_dist, ay / goal_dist
    vx, vy = ax, ay
    for ox, oy, r in obstacles:
        dx, dy = x - ox, y - oy
        centre_dist = math.hypot(dx, dy) or 1e-6
        clearance = centre_dist - (r + inflate)
        if clearance > influence:
            continue
        # Goal is nearer than the obstacle surface -> it is not in the way.
        if goal_dist < centre_dist - r and clearance > 0.15:
            continue
        nx, ny = dx / centre_dist, dy / centre_dist
        heading_into = -(ax * nx + ay * ny)  # > 0 when goal direction hits it
        if heading_into < -0.2 and clearance > 0.15:
            continue
        weight = (influence - max(clearance, 0.0)) / influence  # 0 .. 1
        tx, ty = -ny, nx
        if tx * ax + ty * ay < 0.0:
            tx, ty = -tx, -ty
        push = 2.5 if clearance < 0.15 else 0.6
        vx += weight * (2.0 * tx + push * nx)
        vy += weight * (2.0 * ty + push * ny)
    return math.atan2(vy, vx)


def heading_controller(theta, desired_heading, max_linear, k_angular=5.0,
                       max_angular=5.0, distance_to_goal=None, k_linear=1.5):
    """Unicycle P-controller: returns (linear, angular) velocity commands.

    Forward speed is scaled by cos(heading error) so the turtle turns on the
    spot when it faces away from the goal instead of driving off in an arc.
    """
    err = wrap_angle(desired_heading - theta)
    linear = max_linear * max(0.0, math.cos(err))
    if distance_to_goal is not None:
        linear = min(linear, k_linear * distance_to_goal)
    angular = clamp(k_angular * err, -max_angular, max_angular)
    return linear, angular


def cruise_controller(theta, desired_heading, max_linear, k_angular=5.0,
                      max_angular=5.0, min_fraction=0.35):
    """Like heading_controller but never stops to turn.

    Used by the prey: stopping to pivot hands ground to the predator, so it
    keeps at least ``min_fraction`` of its speed and carves round instead.
    """
    err = wrap_angle(desired_heading - theta)
    scale = min_fraction + (1.0 - min_fraction) * max(0.0, math.cos(err))
    return max_linear * scale, clamp(k_angular * err, -max_angular,
                                     max_angular)


def latched_qos():
    """QoS for one-shot status messages that late subscribers must still see."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


# --------------------------------------------------------------------------
# World construction (spawning turtles / drawing obstacles)
# --------------------------------------------------------------------------
class WorldBuilder:
    """Synchronous wrapper around turtlesim's services, used by spawn nodes."""

    def __init__(self, node):
        import rclpy  # local import keeps the module importable without ROS
        self._rclpy = rclpy
        self.node = node
        self._clients = {}

    def _client(self, srv_type, name):
        if name not in self._clients:
            self._clients[name] = self.node.create_client(srv_type, name)
        return self._clients[name]

    def call(self, srv_type, name, request, timeout=10.0):
        client = self._client(srv_type, name)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service {name} not available')
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self.node, future,
                                               timeout_sec=timeout)
        return future.result()

    def kill(self, name):
        try:
            self.call(Kill, '/kill', Kill.Request(name=name), timeout=3.0)
        except RuntimeError:
            pass

    def spawn(self, name, x, y, theta):
        req = Spawn.Request(x=float(x), y=float(y), theta=float(theta),
                            name=name)
        self.call(Spawn, '/spawn', req)
        self.node.get_logger().info(
            f'spawned {name} at ({x:.2f}, {y:.2f}, {theta:.2f})')

    def set_pen(self, name, r, g, b, width=3, off=False):
        req = SetPen.Request(r=int(r), g=int(g), b=int(b), width=int(width),
                             off=int(bool(off)))
        self.call(SetPen, f'/{name}/set_pen', req)

    def teleport(self, name, x, y, theta=0.0):
        req = TeleportAbsolute.Request(x=float(x), y=float(y),
                                       theta=float(theta))
        self.call(TeleportAbsolute, f'/{name}/teleport_absolute', req)

    def draw_obstacles(self, obstacles, colour=(40, 40, 40), pen_width=9):
        """Paint filled circles for each obstacle using a temporary turtle.

        turtlesim has no notion of obstacles, so they are drawn onto the
        background with a 'painter' turtle that teleports around concentric
        rings, then is removed.  The robots treat the same list of circles as
        no-go zones in software.
        """
        if not obstacles:
            return
        painter = 'obstacle_painter'
        ox, oy, _ = obstacles[0]
        self.spawn(painter, ox, oy, 0.0)
        for ox, oy, radius in obstacles:
            self.set_pen(painter, *colour, width=pen_width, off=True)
            self.teleport(painter, ox + radius, oy)
            time.sleep(0.05)  # let turtlesim process the pen-up move
            self.set_pen(painter, *colour, width=pen_width, off=False)
            ring = radius
            while ring > 0.0:
                steps = max(12, int(ring * 30))
                for i in range(steps + 1):
                    a = 2.0 * math.pi * i / steps
                    self.teleport(painter, ox + ring * math.cos(a),
                                  oy + ring * math.sin(a))
                ring -= 0.15
            self.teleport(painter, ox, oy)
            time.sleep(0.1)  # let queued teleports be drawn before pen-up
        self.set_pen(painter, 0, 0, 0, off=True)
        time.sleep(0.1)
        self.kill(painter)
        self.node.get_logger().info(f'drew {len(obstacles)} obstacles')


def make_rng(seed):
    """Seeded RNG if seed >= 0, otherwise a time-seeded one."""
    return random.Random(seed if seed is not None and seed >= 0 else None)
