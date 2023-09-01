from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
import torch


env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Examine the State and Action Spaces
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

NUM_AGENTS = 2

#agents = [Agent(state_size = state_size, action_size = action_size, random_seed=2) for i in range(NUM_AGENTS)]
agents = Agent(state_size = state_size, action_size = action_size, random_seed=2)

scores_list= [] # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
max_t = 1000
n_episodes = 10000

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
    states = env_info.vector_observations # get the current state (for each agent)
    scores = np.zeros(NUM_AGENTS)                          # initialize the score (for each agent)
    agents.reset()
    #[agent.reset() for agent in agents]
    for t in range(max_t):
        #actions = [agent.act(states[i]) for i, agent in enumerate(agents)] # select an action (for each agent)
        actions = agents.act(states) # select an action (for each agent)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment       
        next_states = env_info.vector_observations       # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                       # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)   
        #[agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], t) for i in range(NUM_AGENTS)] # agent takes a step
        for i in range(NUM_AGENTS):
            agents.step(states[i], actions[i], rewards[i], next_states[i], dones[i], t) # agents take a step
        states = next_states                             # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    scores_window.append(max(scores))       # save most recent score
    scores_list.append(max(scores))          # save most recent score
   
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, max(scores)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if len(scores_window) == 100 and np.mean(scores_window) >= 0.5: 
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        torch.save(agents.actor_local.state_dict(), 'model_actor.pth')
        torch.save(agents.critic_local.state_dict(), 'model_critic.pth')
        break
env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_list)), scores_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()