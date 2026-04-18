import sys
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner

def test_can_with_obs_image_runner():
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    task_cfg_path = os.path.join(
        ROOT_DIR,
        'diffusion_policy',
        'config',
        'task',
        'can_with_obs_image_abs.yaml',
    )
    task_cfg = OmegaConf.load(task_cfg_path)
    cfg = OmegaConf.create({
        'task': task_cfg,
        'n_obs_steps': 1,
        'n_action_steps': 1,
        'past_action_visible': False,
    })
    runner_cfg = cfg['task']['env_runner']
    runner_cfg['n_train'] = 1
    runner_cfg['n_test'] = 1
    runner_cfg['n_envs'] = 2
    del runner_cfg['_target_']
    runner = RobomimicImageRunner(
        **runner_cfg, 
        output_dir='/tmp/test_can_with_obs_image_runner')

    env = runner.env
    env.seed(seeds=runner.env_seeds)
    obs = env.reset()
    assert obs is not None
    action = np.zeros(
        (len(runner.env_fns), runner.n_action_steps, cfg['task']['shape_meta']['action']['shape'][0]),
        dtype=np.float32,
    )
    if runner.abs_action:
        action = runner.undo_transform_action(action)
    for i in range(10):
        _ = env.step(action)

    imgs = env.render()
    assert imgs is not None

if __name__ == '__main__':
    test_can_with_obs_image_runner()
