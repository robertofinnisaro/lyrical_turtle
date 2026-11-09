"""Offline trainer for the Task 4 prey (no turtlesim needed).

Runs thousands of games in a turtlesim-equivalent kinematic simulator using
exactly the predator and prey code the ROS nodes use, then saves the
Q-table where the prey node will load it.

Every tenth of the run the greedy policy is scored on a fixed *validation*
set of start positions and the best table so far is kept (tabular
Q-learning against several moving predators is noisy).  The final numbers
are reported on a separate *test* set the selection never saw.

    ros2 run task4 train_prey --episodes 20000 --out prey-kb.json
"""

import argparse
import random
import time

from .core import PreyAgent, run_episode


def evaluate(agent, policy, episodes, seed):
    rng = random.Random(seed)
    survived = total = 0.0
    for _ in range(episodes):
        ok, t = run_episode(agent, rng, policy, learn=False)
        survived += ok
        total += t
    return survived / episodes, total / episodes


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--out', default='prey-kb.json')
    parser.add_argument('--resume', action='store_true',
                        help='continue training an existing table')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eval-episodes', type=int, default=300)
    opts, _ = parser.parse_known_args(args)

    agent = PreyAgent(epsilon=0.3, seed=opts.seed)
    if opts.resume and agent.load(opts.out):
        print(f'resuming from {opts.out} ({agent.episodes} episodes)')
    rng = random.Random(opts.seed)

    validation, test = 999, 12345
    start = time.time()
    report = max(1, opts.episodes // 10)
    best_score, best_q = -1.0, None
    for i in range(1, opts.episodes + 1):
        run_episode(agent, rng, 'q', learn=True)
        if i % report == 0:
            surv, alive = evaluate(agent, 'q', opts.eval_episodes,
                                   validation)
            marker = ''
            if alive > best_score:
                best_score = alive
                best_q = ({k: list(v) for k, v in agent.q.items()},
                          agent.episodes)
                marker = '  <- best so far'
            print(f'{i:6d} episodes  eps={agent.epsilon:.3f}  '
                  f'alpha={agent.alpha:.3f}  states={len(agent.q):4d}  '
                  f'survival {surv:.0%}  mean time alive {alive:.1f} s  '
                  f'({time.time() - start:.0f} s){marker}')
    if best_q is not None:
        agent.q = best_q[0]
    agent.save(opts.out)
    print(f'saved best Q-table to {opts.out}')

    print('\ntest set (unseen start positions):')
    for label, policy in (('random escape heading', 'random'),
                          ('flee nearest predator', 'away'),
                          ('learnt Q-policy', 'q')):
        surv, alive = evaluate(agent, policy, opts.eval_episodes, test)
        print(f'  {label:24s} survival {surv:4.0%}   mean time alive '
              f'{alive:5.1f} s')


if __name__ == '__main__':
    main()
