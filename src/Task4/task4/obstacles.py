"""Static obstacles shared by every node: (centre_x, centre_y, radius) in m.

turtlesim cannot model obstacles, so the spawn node paints these circles on
the background and every robot treats them as no-go zones in software.
"""

OBSTACLES = [
    (3.0, 8.0, 0.9),
    (8.2, 7.8, 0.8),
    (7.8, 3.0, 1.0),
    (2.8, 3.0, 0.7),
    (5.6, 5.5, 0.6),
]
