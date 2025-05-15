import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from aquacropgymnasium.env import Wheat
#from ppo import Agent
from ppo2 import PPO

output_dir = "./train_output"
os.makedirs(output_dir, exist_ok=True)

timestep_targets = [500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]

class RewardLogger:
    def __init__(self, agent_name, output_dir, num_intervals=100):
        self.agent_name = agent_name
        self.output_dir = output_dir
        self.num_intervals = num_intervals
        self.episode_rewards = []

    def log(self, reward):
        self.episode_rewards.append(reward)

    def finalize(self):
        rewards = np.array(self.episode_rewards)
        avg = np.mean(rewards)
        std = np.std(rewards)
        print(f"[{self.agent_name}] Final mean reward: {avg:.2f}, std: {std:.2f}")
        self._plot()

    def _plot(self):
        x = np.arange(len(self.episode_rewards))
        plt.figure(figsize=(10, 5))
        plt.plot(x, self.episode_rewards, marker='o')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"reward_plot_{self.agent_name}.png"), dpi=300)
        plt.close()

for train_timesteps in timestep_targets:
    wandb.init(
        project="ppo-wheat-irrigation",
        name=f"ppo_agent_{train_timesteps}",
        config={
            "train_timesteps": train_timesteps,
            "gamma": 0.98,
            "alpha": 6.34e-04,
            "gae_lambda": 0.95,
            "policy_clip": 0.22,
            "batch_size": 512,
            "n_epochs": 23
        }
    )

    env = Wheat(mode="train", year1=1982, year2=2007)
    agent = PPO(
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        gamma=0.98,
        alpha=6.34e-04,
        gae_lambda=0.95,
        policy_clip=0.22,
        batch_size=512,
        N=2048,
        n_epochs=23
    )
    logger = RewardLogger(agent_name=f"ppo_agent{train_timesteps}", output_dir=output_dir)

    total_timesteps = 0
    episode = 0

    while total_timesteps < train_timesteps:
        obs, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action, prob, val = agent.choose_action(obs)
            obs_, reward, done, _, _ = env.step(action)
            agent.remember(obs, action, prob, val, reward, done)
            score += reward
            steps += 1
            obs = obs_

        agent.learn()
        logger.log(score)
        total_timesteps += steps

        wandb.log({
            "Episode": episode,
            "Episode_Reward": score,
            "Steps_in_Episode": steps,
            "Total_Timesteps": total_timesteps
        })
        print(f"[{train_timesteps}] Episode {episode} | Steps: {steps} | Reward: {score:.2f} | Timesteps: {total_timesteps}")
        episode += 1

    agent.save_model(path=os.path.join(output_dir, f"ppo_agent_{train_timesteps}.pth"))
    logger.finalize()
    wandb.finish()
