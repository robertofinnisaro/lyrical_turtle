# ACW1 – Turtlesim predator / prey (ROS 2)

A colcon workspace with one `ament_python` package per task. It targets ROS 2
Lyrical and should also build on Humble, because the code imports turtlesim's
interfaces from `turtlesim_msgs` (Jazzy and later) and falls back to
`turtlesim` (Humble).

```
src/Task1  -> package task1   single predator, static prey
src/Task2  -> package task2   single predator, fleeing prey, 60 s limit
src/Task3  -> package task3   learning predator (predator-kb.xml) + obstacles
src/Task4  -> package task4   3 predators vs Q-learning prey + obstacles
tools/     -> offline renderer used to make the demo videos (not a ROS package)
```

## Build (WSL2 on the lab PCs)

```bash
source /opt/ros/lyrical/setup.bash        # or /opt/ros/humble/setup.bash
git clone <this repo> ~/acw_ws
cd ~/acw_ws
colcon build --symlink-install
source install/setup.bash
```

WSLg provides the display for the turtlesim window. No extra Python
dependencies are needed beyond a standard ROS desktop install.

## Run

Each launch file starts turtlesim, a spawn node, the robots and (for Tasks 2–4)
a referee. When a game ends the robots stop and the turtlesim window stays open
on the final frame. Press Ctrl+C to close it. Add `seed:=N` to any launch to
reproduce the same start positions.

```bash
ros2 launch task1 task1.launch.py
ros2 launch task2 task2.launch.py
ros2 launch task3 task3.launch.py seed:=7       # run 3+ times from the same directory
ros2 launch task4 task4.launch.py
```

### Task 3: learning across runs
`predator-kb.xml` is written to the directory you launch from, or to
`kb_path:=...` if you set it. Trial 1 has no knowledge and uses plain pursuit.
Every later trial loads all previous trials and chases a virtual target (the
predicted prey position). The virtual target is shown as an extra turtle called
`virtual_target`; turn it off with `show_virtual_target:=false`. For a baseline
comparison, run with `use_kb:=false`. Delete the XML file to start over.

### Task 4: learning prey
The prey uses tabular Q-learning. It loads `prey-kb.json` from the launch
directory, or the pre-trained table shipped in the package if that file doesn't
exist (`use_pretrained:=false` starts from scratch). It keeps learning online
and saves after every game. To retrain offline (about 4 minutes, no turtlesim
needed, using the same code as the nodes):

```bash
ros2 run task4 train_prey --episodes 30000 --out prey-kb.json
```

## Design summary

| | Predator | Prey | Ending |
|---|---|---|---|
| 1 | P-controller on Euclidean distance and bearing from `/prey_location` | static, publishes `/prey_location` | predator stops within 0.5 m |
| 2 | pure pursuit, 1.5 m/s, publishes `/predator_location` | scores 32 candidate moves (x, y, θ): distance from the predicted predator position, quadratic wall penalty with look-ahead, commitment term; 1.45 m/s | referee: capture < 0.5 m or 60 s |
| 3 | records both positions every 0.5 s turn into `predator-kb.xml`; k-NN prediction in the prey's frame gives a virtual intercept target; vector-field obstacle avoidance; 1.7 m/s | as Task 2, plus obstacle veto and penalty | referee |
| 4 | three independent nodes, lead factors 0 / 0.6 / 1.0 (chaser to interceptor), obstacle and teammate avoidance; 1.5 m/s | Q-learning: 324 states × 8 relative escape headings, a safety shield masks unsafe moves; 1.9 m/s | referee: any predator < 0.5 m or 60 s |

turtlesim has no obstacles, so the spawn node paints five filled circles with a
temporary "painter" turtle. All robots treat the same list (`obstacles.py`) as
no-go zones in software.

## Offline results (turtlesim-equivalent kinematic simulator)

* **Task 3:** knowledge base built from 10 trials, then tested on 30 unseen start
  positions. Mean capture time was 21.2 s with plain pursuit and 12.1 s with the
  knowledge base. The knowledge-base predictor was faster in 30 of 30 cases, with
  0 obstacle collisions.
* **Task 4:** 300 unseen starts.

  | prey policy | survives 60 s | mean time alive |
  |---|---|---|
  | random heading | 0 % | 3.3 s |
  | flee nearest predator | 6 % | 10.1 s |
  | learnt Q-policy | 20 % | 19.1 s |

These numbers come from the offline simulator, not from turtlesim. Timing in
turtlesim will vary somewhat.
