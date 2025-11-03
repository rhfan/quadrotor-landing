import argparse
import os
import pickle
from importlib import metadata

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from quadenv import quadEnv
from quadenv2 import QuadEnv_ocp
from quadenv3 import QuadEnv_ocpv2
from quadenv5 import QuadEnv_polyv3
def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": 1,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="test_ocpv5_75v3")
    parser.add_argument("--ckpt", type=int, default=400)
    parser.add_argument("--record", action="store_true", default=True)
    args = parser.parse_args()

    gs.init()
    log_dir = f"logs/{args.exp_name}"
    env=QuadEnv_polyv3(1,True)
    train_cfg=get_train_cfg(args.exp_name)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    max_sim_step = int(15 * 60)
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="logs/test_ocpv5_75v3.mp4", fps=60)
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
