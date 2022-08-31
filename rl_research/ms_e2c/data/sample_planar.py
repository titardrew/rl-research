import numpy as np
import os
from os import path
from tqdm import trange
import json
from datetime import datetime
import argparse
from PIL import Image, ImageDraw

np.random.seed(1)

width, height = 40, 40
obstacles_center = np.array([[20.5, 5.5], [20.5, 12.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])

r_overlap = 0.5 # agent cannot be in any rectangular area with obstacles as centers and half-width = 0.5
r = 2.5 # radius of the obstacles when rendered
rw = 3 # robot half-width
rw_rendered = 2 # robot half-width when rendered
max_step_len = 3
env_path = os.path.dirname(os.path.abspath(__file__))
env = np.load(env_path + '/env.npy')

def get_pixel_location(s):
    # return the location of agent when rendered
    center_x, center_y = int(round(s[0])), int(round(s[1]))
    top = center_x - rw_rendered
    bottom = center_x + rw_rendered
    left = center_y - rw_rendered
    right = center_y + rw_rendered
    return top, bottom, left, right

def render(s):
    top, bottom, left, right = get_pixel_location(s)
    x = np.copy(env)
    x[top:bottom, left:right] = 1.  # robot is white on black background
    return x

def is_valid(s, u, s_next, epsilon = 0.1):
    # if the difference between the action and the actual distance between x and x_next are in range(0,epsilon)
    top, bottom, left, right = get_pixel_location(s)
    top_next, bottom_next, left_next, right_next = get_pixel_location(s_next)
    x_diff = np.array([top_next - top, left_next - left], dtype=np.float)
    return (not np.sqrt(np.sum((x_diff - u)**2)) > epsilon)

def is_colliding(s):
    """
    :param s: the continuous coordinate (x, y) of the agent center
    :return: if agent body overlaps with obstacles
    """
    if np.any([s - rw < 0, s + rw > height]):
        return True
    x, y = s[0], s[1]
    for obs in obstacles_center:
        if np.abs(obs[0] - x) <= r_overlap and np.abs(obs[1] - y) <= r_overlap:
            return True
    return False

def random_step(s):
    # draw a random step until it doesn't collidie with the obstacles
    while True:
        u = np.random.uniform(low = -max_step_len, high = max_step_len, size = 2)
        s_next = s + u
        if (not is_colliding(s_next) and is_valid(s, u, s_next)):
            return u, s_next


def sample(sample_size):
    state_samples, obs_samples = sample_n(sample_size, 1)
    return state_samples, obs_samples


def sample_n(sample_size, n_steps=1):
    """
    return [(s, u, s_next)]
    """
    assert n_steps > 0

    state_samples = []
    for i in trange(sample_size, desc = 'Sampling data'):
        # Selecting initial_state
        while True:
            s_x = np.random.uniform(low = rw, high = height - rw)
            s_y = np.random.uniform(low = rw, high = width - rw)
            s = np.array([s_x, s_y])
            if not is_colliding(s):
                break

        trajectory = [s]
        for i_step in range(n_steps):
            u, s = random_step(s)
            trajectory += [u, s]
        state_samples.append(tuple(trajectory))
        

    # To obs (render)
    obs_samples = []
    for trajectory in state_samples:
        obs_trajectory = []
        for i, el in enumerate(trajectory):
            if i % 2 == 0:
                el = render(el)
            obs_trajectory.append(el)
        obs_samples.append(tuple(obs_trajectory))
        

    return state_samples, obs_samples


def write_to_file(sample_size, output_dir_base = './data/planar', n_steps=1):
    """
    write [(x, u, x_next, u_next ...)] to output dir
    """
    assert n_steps > 0

    output_dir = "{}_{}".format(output_dir_base, n_steps)

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    state_samples, obs_samples = sample_n(sample_size, n_steps=n_steps)

    samples = []
    for i, trajectory in enumerate(obs_samples):
        obs_file_list = []
        state_list = []
        control_list = []
        assert len(trajectory) == 2*n_steps + 1, len(trajectory)
        for el_idx, el in enumerate(trajectory):
            if el_idx % 2 == 0:  # is an observation
                state = state_samples[i][el_idx]
                obs = el
                obs_idx = el_idx // 2  # obs0 - 0, obs1 - 2, obs2 - 4, ...
                obs_file = 'obs-{:02d}-{:05d}.png'.format(obs_idx, i)
                Image.fromarray(obs * 255.).convert('L').save(path.join(output_dir, obs_file))

                obs_file_list.append(obs_file)
                state_list.append(state.tolist())

            else:  # is a control (action)
                u = el
                control_list.append(u.tolist())

        samples.append({
            'states': state_list,
            'observation_files': obs_file_list,
            'controls': control_list,
        })


    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'num_steps': n_steps,
                    'max_distance': max_step_len,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size
    n_steps = args.n_steps

    write_to_file(sample_size=sample_size, n_steps=n_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--n_steps', required=False, default=1, type=int, help='the number of steps in a sample.')

    args = parser.parse_args()

    main(args)
