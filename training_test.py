import getopt
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool

from observation_utils import normalize_observation
from dddqn_policy import DDDQNPolicy

np.random.seed(1)
torch.manual_seed(1)

observation_tree_depth = 2
NUM_AGENT = 15
NUM_CITIES = 5

class Parameters:
    def __init__(self):
        self.hidden_size = 256
        self.buffer_size = 10000
        self.batch_size = 128
        self.update_every = 8
        self.learning_rate = 0.0005
        self.tau = 0.001
        self.gamma = 0.99
        self.buffer_min_size = 0
        self.use_gpu = False

def create_env():
    nAgents = NUM_AGENT
    n_cities = NUM_CITIES
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    env = RailEnv(
        width=30,
        height=30,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=nAgents,
        obs_builder_object=TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=ShortestPathPredictorForRailEnv())
    )
    return env

def training_example(sleep_for_animation, do_rendering):

    np.random.seed(1)
    env = create_env()
    env.reset()

    env_renderer = None
    if do_rendering:
        env_renderer = RenderTool(env)
    
    # Observation parameters
    observation_radius = 3

    # Training parameters
    n_episodes = 20
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.997

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes
    #print("Number of Features per node: ", n_features_per_node)

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Max number of steps per episode
    max_steps = int(4 * 2 * (env.height + env.width + (NUM_AGENT / NUM_CITIES)))
    # max_steps = env._max_episode_steps

    action_count = [0] * action_size
    action_dict = dict()

    scores = []
    completions = []

    parameters = Parameters()
    agent_obs = [None] * NUM_AGENT
    agent_prev_obs = [None] * NUM_AGENT
    agent_prev_action = [2] * NUM_AGENT
    update_values = [False] * NUM_AGENT

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, action_size, parameters)

    for episode_idx in range(n_episodes + 1):
        obs, info = env.reset()
        # print("INFO: ", info)

        if env_renderer is not None:
            env_renderer.reset()

        score = 0
        # actions_taken = []

        # Build initial agent-specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                    # actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            if env_renderer is not None:
                env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

            # Update replay buffer and train agent
            for agent in env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]
                # print(f"Agent {agent}: Reward {all_rewards[agent]}, Observation {agent_obs[agent]}")
                # Preprocess the new observations
                if next_obs[agent]:
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                score += all_rewards[agent]

            if done['__all__']:
                break
        print('Episode Nr. {}\t Score = {}'.format(episode_idx, score))

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)
        eps_start = max(eps_end, eps_start * eps_decay)
        #nb_steps.append(final_step)
        print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(score), np.mean(completion) * 100.0))
    
    if env_renderer is not None:
        env_renderer.close_window()


def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", "do_rendering=", ""])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    sleep_for_animation = True
    do_rendering = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        elif o in ("--do_rendering"):
            do_rendering = str2bool(a)
        else:
            assert False, "unhandled option"

    training_example(sleep_for_animation, do_rendering)

if __name__ == '__main__':
    if 'argv' in globals():
        main(argv)
    else:
        main(sys.argv[1:])