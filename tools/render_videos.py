import math, os, random, sys, tempfile
from render import *  # noqa: F401,F403

for t in (1, 2, 3, 4):
    sys.path.insert(0, f'{SRC}/Task{t}')
from task1.common import heading_controller, random_free_point, make_rng, distance
from task2.prey_logic import plan_escape as plan2
from task3.prey_logic import plan_escape as plan3
from task3.common import avoidance_heading, cruise_controller
from task3.knowledge_base import KnowledgeBase
from task3.obstacles import OBSTACLES
from task4 import core as C

PRED_SHELL, PREY_SHELL = (190, 120, 50), (120, 200, 230)


# ---------------------------------------------------------------- Task 1
def sim_task1(seed, record=None):
    rng = make_rng(seed)
    a = random_free_point(rng)
    b = random_free_point(rng, others=[a], min_separation=4.0)
    P = T('predator', *a, rng.uniform(-3, 3), PRED_SHELL, (255, 140, 0))
    Q = T('prey', *b, rng.uniform(-3, 3), PREY_SHELL, None)
    t = 0.0
    while True:
        dx, dy = Q.x - P.x, Q.y - P.y
        d, ang = math.hypot(dx, dy), math.atan2(dy, dx)
        if record:
            record(P, Q, t, d, ang, False)
        if d < 0.5:
            if record:
                record(P, Q, t, d, ang, True)
            return t
        P.step(*heading_controller(P.th, ang, 2.0, distance_to_goal=d))
        t += DT


def video_task1():
    v = Video(f'{OUT}/task1_static_target.mp4')
    title_card(v, ['Task 1', 'Single robot, static target',
                   'spawn_turtles kills turtle1 and spawns "predator" and '
                   '"prey" at random positions. The prey publishes its pose '
                   'on /prey_location; the predator subscribes, computes '
                   'Euclidean distance and bearing, and drives to it.'], 5)
    caps = [(0, 'The prey is static and publishes its coordinates on '
                '/prey_location (10 Hz). The predator subscribes to it and to '
                'its own /predator/pose.'),
            (3, 'Each 20 Hz control tick: distance = hypot(dx, dy), bearing '
                '= atan2(dy, dx). A proportional controller turns towards the '
                'bearing; forward speed is scaled by cos(heading error) so it '
                'turns on the spot first, then drives in.'),
            (7, 'When the distance falls below 0.5 m the predator publishes '
                '"capture" on /game_status, stops, and the simulation ends.')]
    for run, seed in enumerate((11, 4)):
        def rec(P, Q, t, d, ang, final):
            hdr = f'Task 1 - run {run + 1}   t = {t:4.1f} s'
            st = f'distance {d:5.2f} m   bearing {math.degrees(ang):5.0f} deg'
            img = frame([Q, P], [], hdr, st, caption_at(caps, t),
                        banner=f'CAPTURED in {t:.1f} s' if final else None)
            v.write(img, 3 * FPS if final else 1)
        sim_task1(seed, rec)
    return v.close()


# ---------------------------------------------------------------- Task 2/3
def sim_chase(seed, obstacles, planner, pred_speed, prey_speed, threat_step,
              kb=None, record=None, kb_record=True):
    rng = make_rng(seed)
    a = random_free_point(rng, obstacles)
    b = random_free_point(rng, obstacles, others=[a], min_separation=4.0)
    P = T('predator', *a, rng.uniform(-3, 3), PRED_SHELL, (255, 140, 0))
    Q = T('prey', *b, rng.uniform(-3, 3), PREY_SHELL, (255, 255, 255))
    move = None
    moves = []
    target = (Q.x, Q.y)
    conf = 0.0
    t, i = 0.0, 0
    turn = round(0.5 / DT)
    learning = kb is not None and kb.num_samples > 0
    result = 'escape'
    while True:
        d = distance(P.x, P.y, Q.x, Q.y)
        if d < 0.5:
            result = 'capture'
        if d < 0.5 or t >= 60.0:
            if record:
                record(P, Q, t, d, target, conf, result)
            break
        if i % turn == 0:
            moves.append((P.x, P.y, Q.x, Q.y))
        if i % 6 == 0:
            move, _ = planner(Q.x, Q.y, Q.th, [(P.x, P.y)], obstacles,
                              threat_step=threat_step,
                              previous=move[2] if move else None)
        if learning and i % 8 == 0 and moves:
            last = moves[-1]
            tx, ty, conf, _ = kb.predict(P.x, P.y, Q.x, Q.y, Q.x - last[2],
                                         Q.y - last[3], pred_speed * 0.5)
            if d < 1.0:
                tx, ty = Q.x, Q.y
            target = (tx, ty)
        elif not learning:
            target = (Q.x, Q.y)
        lin, ang = cruise_controller(Q.th, move[2], prey_speed)
        if not obstacles and d > 6.0:
            lin *= 0.6
        Q.step(lin, ang)
        h = avoidance_heading(P.x, P.y, *target, obstacles)
        P.step(*heading_controller(P.th, h, pred_speed))
        if record:
            record(P, Q, t, d, target, conf, None)
        t += DT
        i += 1
    moves.append((P.x, P.y, Q.x, Q.y))
    if kb is not None and kb_record:
        kb.append_trial(moves, result)
    return t, result


def video_task2():
    v = Video(f'{OUT}/task2_moving_target.mp4')
    title_card(v, ['Task 2', 'Single robot, moving target',
                   'The prey subscribes to /predator_location and every 0.2 s '
                   'scores 32 candidate moves (x, y, theta), picking the one '
                   'that ends farthest from where the predator will be, while '
                   'penalising arena edges so it turns BEFORE the wall. A '
                   'referee node stops the game on capture or after 60 s.'], 6)
    caps = [(0, 'Predator (orange trail) runs pure pursuit on /prey_location '
                'at 1.5 m/s. Prey (white trail) is slightly slower (1.45 m/s) '
                'so it must out-think, not out-run, the predator.'),
            (4, 'Candidate scoring: distance from the predator\'s predicted '
                'position, minus a quadratic wall penalty inside 1.8 m of an '
                'edge and a look-ahead wall check, minus a commitment penalty '
                'that stops it dithering between two equal escape routes.'),
            (12, 'Near a wall the prey curves along it rather than stopping; '
                 'the predator cuts the corner, which is how most captures '
                 'happen in a bounded arena.'),
            (25, 'The referee checks ground-truth poses at 20 Hz: capture '
                 'distance 0.5 m, time limit 60 s; it then publishes '
                 '/game_status and zeroes every cmd_vel.')]
    for run, seed in enumerate(SEEDS2):
        def rec(P, Q, t, d, target, conf, result):
            hdr = f'Task 2 - run {run + 1}   t = {t:4.1f} / 60 s'
            st = f'predator-prey distance {d:5.2f} m'
            banner = None
            if result == 'capture':
                banner = f'CAPTURED after {t:.1f} s'
            elif result:
                banner = 'PREY ESCAPED (60 s)'
            img = frame([Q, P], [], hdr, st, caption_at(caps, t),
                        banner=banner)
            v.write(img, 3 * FPS if result else 1)
        sim_chase(seed, (), plan2, 1.5, 1.45, 1.25, record=rec)
    return v.close()


def video_task3():
    v = Video(f'{OUT}/task3_learning_predator.mp4')
    title_card(v, ['Task 3', 'Predator with learning + obstacles',
                   'Every 0.5 s turn the predator records both positions; at '
                   'the end the trial is appended to predator-kb.xml. On the '
                   'next run it loads the file and chases a VIRTUAL TARGET: '
                   'the prey position predicted by k-nearest-neighbour '
                   'regression over every past turn. Same start (seed) each '
                   'trial so they can be compared.'], 7)
    path = tempfile.mktemp(suffix='.xml')
    kb = KnowledgeBase(path, OBSTACLES)
    times = []
    for trial in (1, 2, 3):
        kb.load()
        caps1 = [(0, 'Trial 1: predator-kb.xml does not exist yet, so the '
                     'predator chases the prey\'s live position. Grey circles '
                     'are obstacles painted by a temporary turtle; both robots '
                     'treat them as no-go zones (vector-field avoidance for '
                     'the predator, vetoed moves for the prey).'),
                 (12, 'The prey uses the obstacles to its advantage, looping '
                      'round them while the predator follows behind.')]
        capsn = [(0, f'Trial {trial}: loaded {len(kb.trials)} previous '
                     f'trial(s), {kb.num_samples} learnt situations. Red ring '
                     '= virtual target. Situations are described in the '
                     'prey\'s own frame (predator offset, speed, free space '
                     'ahead/left/right) so learning transfers between '
                     'directions.'),
                 (10, 'The predator aims where the prey is likely to be when '
                      'it arrives (intercept horizon chosen from its own '
                      'travel time), cutting corners instead of tail-chasing.')]
        caps = caps1 if trial == 1 else capsn
        def extra_for(target, P, show):
            def extra(d):
                if not show:
                    return
                x, y = px(*target)
                d.line([px(P.x, P.y), (x, y)], fill=(255, 80, 80), width=1)
                d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 40, 40),
                          width=3)
            return extra
        def rec(P, Q, t, d, target, conf, result, trial=trial):
            hdr = f'Task 3 - trial {trial}   t = {t:4.1f} / 60 s'
            prev = '' if not times else '   previous: ' + ', '.join(
                f'{x:.1f}s' for x in times)
            st = f'distance {d:5.2f} m{prev}'
            banner = None
            if result == 'capture':
                banner = f'CAPTURED after {t:.1f} s'
            elif result:
                banner = 'PREY ESCAPED (60 s)'
            img = frame([Q, P], OBSTACLES, hdr, st, caption_at(caps, t),
                        extra=extra_for(target, P, trial > 1),
                        banner=banner)
            v.write(img, 3 * FPS if result else 1)
        t, res = sim_chase(SEED3, OBSTACLES, plan3, 1.7, 1.45, 1.45, kb=kb,
                           record=rec)
        times.append(t)
    os.remove(path)
    print('task3 trial times', times)
    return v.close()


# ---------------------------------------------------------------- Task 4
PEN4 = {'predator1': (255, 80, 80), 'predator2': (255, 170, 0),
        'predator3': (255, 0, 200)}


def sim_task4(seed, agent, policy, record=None):
    rng = random.Random(seed)
    prey0, preds0 = C.random_world(rng)
    Q = T('prey', prey0.x, prey0.y, prey0.theta, PREY_SHELL, (255, 255, 255))
    Ps = [T(n, p.x, p.y, p.theta, PRED_SHELL, PEN4[n])
          for n, p in zip(C.PREDATOR_NAMES, preds0)]
    t, i = 0.0, 0
    heading = Q.th
    per = round(C.DECISION_PERIOD / DT)
    action = 0
    while True:
        pxy = [(p.x, p.y) for p in Ps]
        dmin = min(distance(Q.x, Q.y, *p) for p in pxy)
        if dmin < 0.5 or t >= 60:
            if record:
                record(Q, Ps, t, dmin, action, 'capture' if dmin < 0.5
                       else 'escape')
            return t, dmin < 0.5
        if i % per == 0:
            away = agent.away_angle((Q.x, Q.y), pxy)
            state = agent.encode((Q.x, Q.y), pxy)
            legal = agent.legal_actions((Q.x, Q.y), away)
            action = (agent.choose(state, legal, explore=False)
                      if policy == 'q' else legal[0])
            heading = agent.action_heading(action, away)
        lin, ang = cruise_controller(Q.th, heading, C.PREY_SPEED)
        vx, vy = Q.v * math.cos(Q.th), Q.v * math.sin(Q.th)
        cmds = [C.predator_command(p.x, p.y, p.th, Q.x, Q.y, vx, vy,
                                   [q for j, q in enumerate(pxy) if j != k],
                                   C.PREDATOR_LEAD[p.name])
                for k, p in enumerate(Ps)]
        Q.step(lin, ang)
        for p, c in zip(Ps, cmds):
            p.step(*c)
        if record:
            record(Q, Ps, t, dmin, action, None)
        t += DT
        i += 1


def video_task4():
    v = Video(f'{OUT}/task4_multi_robot_learning_prey.mp4')
    title_card(v, ['Task 4', '3 predators vs a learning prey + obstacles',
                   'predator1 chases, predator2 and predator3 intercept '
                   '(lead 0.6 / 1.0); each acts independently. The prey uses '
                   'tabular Q-learning (prey-kb.json) over 8 escape headings '
                   'relative to the nearest predator, with a safety shield '
                   'that masks moves into walls or obstacles. Same start '
                   'positions in both runs.'], 7)
    agent = C.PreyAgent()
    agent.load(f'{SRC}/Task4/config/prey-kb-pretrained.json')
    names = ['away', '+45', '-45', '+90', '-90', '+135', '-135', 'toward']
    runs = [('away', 'Run A - untrained prey: "flee the nearest predator"',
             [(0, 'Baseline: with an empty Q-table every action ties and '
                  'the prey just runs directly away from whichever predator '
                  'is nearest. It flips between threats and gets herded into '
                  'walls and corners.')]),
            ('q', 'Run B - prey after Q-learning',
             [(0, 'Same start positions, prey now uses the learnt Q-table '
                  '(30 000 offline episodes using the exact node code, then '
                  'keeps learning online each run). State: distance to the '
                  'nearest predator, bearing of the second predator, bearing '
                  'of the nearest wall/obstacle.'),
              (12, 'What it learnt here: loop the outer corridor round the '
                   'obstacles so all three predators end up trailing behind '
                   'it, instead of reversing into them. Reward: +0.1 per '
                   'step, +distance gained, -10 if caught, +10 for 60 s.'),
              (35, 'Weakness exposed: independent predators follow each '
                   'other in a train. Cooperative herding would beat this '
                   'prey - a good point for the critique.')])]
    for policy, label, caps in runs:
        def rec(Q, Ps, t, dmin, action, result, label=label, caps=caps):
            hdr = f'Task 4 - {label.split(" - ")[0]}   t = {t:4.1f} / 60 s'
            st = (f'{label.split(" - ")[1][:34]}   nearest {dmin:4.2f} m   '
                  f'action: {names[action]}')
            banner = None
            if result == 'capture':
                banner = f'CAUGHT after {t:.1f} s'
            elif result:
                banner = 'PREY ESCAPED (60 s)'
            img = frame([Q] + Ps, OBSTACLES, hdr, st, caption_at(caps, t),
                        banner=banner)
            v.write(img, 3 * FPS if result else 1)
        sim_task4(SEED4, agent, policy, rec)
    return v.close()


SEEDS2 = []
SEED3 = None
SEED4 = None

if __name__ == '__main__':
    which = sys.argv[1:]
    if '1' in which:
        print('task1', video_task1())
    if '2' in which:
        SEEDS2 = [int(s) for s in os.environ['SEEDS2'].split(',')]
        print('task2', video_task2())
    if '3' in which:
        SEED3 = int(os.environ['SEED3'])
        print('task3', video_task3())
    if '4' in which:
        SEED4 = int(os.environ['SEED4'])
        print('task4', video_task4())
