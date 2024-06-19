import argparse
import os
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback

from mario_net import create_mario_env, MarioNet

save_dir = Path('./model')
reward_log_path = (save_dir / 'reward_log.csv')


# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000

# Model Param
CHECK_FREQ_NUMB = 10000
TOTAL_TIMESTEP_NUMB = 5000000
LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = (self.save_path / 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

                state = env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', TOTAL_TIMESTEP_NUMB)
            print('average reward:', (sum(total_reward) / EPISODE_NUMBERS),
                  'average time:', (sum(total_time) / EPISODE_NUMBERS),
                  'best_reward:', best_reward)

            with open(reward_log_path, 'a') as f:
                print(self.n_calls, ',', sum(total_reward) / EPISODE_NUMBERS, ',', best_reward, file=f)

        return True


callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir)

def training(sb3_class):
    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # model = PPO('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=save_dir, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
    #             batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, ent_coef=ENT_COEF)
    model = sb3_class('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=save_dir)

    model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback)


def replay(sb3_class):
    reward_log = pd.read_csv(reward_log_path.absolute(), index_col='timesteps')
    best_epoch = reward_log['reward'].idxmax()
    print('best epoch:', best_epoch)

    best_model_path = save_dir / 'best_model_{}'.format(best_epoch)
    model = sb3_class.load(best_model_path)

    state = env.reset()
    done = False
    plays = 0
    wins = 0
    while plays < 100:
        action, _ = model.predict(state)
        # e = envs[i]
        # i = (i + 1) % skips
        state, reward, done, info = env.step(action)
        env0.render()
        time.sleep(1.0 / 50 * 4)
        if done:
            state = env.reset()
            # for i in range(0, skips):
            #     e = envs[i]
            #     state = e.reset()
            if info[0]["flag_get"]:
                wins += 1
            plays += 1
    print("Model win rate: " + str(wins) + "%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('--sb3_algo', default='PPO', help='StableBaseline3 RL algorithm i.e. A2C, DQN, PPO')
    parser.add_argument('--movement', default='simple', help='simple or complex')
    parser.add_argument('--world', help='world', type=int, default=1)
    parser.add_argument('--stage', help='stage', type=int, default=1)
    parser.add_argument('--replay', help='Replay mode', action='store_true')
    parser.add_argument('--plot', help='Plot mode', action='store_true')
    args = parser.parse_args()

    env_name = f'SuperMarioBros-{args.world}-{args.stage}-v0'
    movements = dict(simple=SIMPLE_MOVEMENT, complex=COMPLEX_MOVEMENT)
    env, env0 = create_mario_env(env_name, movements[args.movement])

    print(f'RL Algorithm: {args.sb3_algo}')
    try:
        sb3_class = getattr(stable_baselines3, args.sb3_algo.upper())
    except:
        print(f'Invalid algorithm: {args.sb3_algo}')
        exit(1)

    if args.replay:
        replay(sb3_class)
    elif args.plot:
        reward_log = pd.read_csv(reward_log_path, index_col='timesteps')
        reward_log.plot()
        plt.show()
    else:
        shutil.rmtree(save_dir, ignore_errors=True)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        with open(reward_log_path, 'a') as f:
            print('timesteps,reward,best_reward', file=f)

        training(sb3_class)
