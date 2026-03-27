import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

st.set_page_config(page_title="Warehouse Robot Q-Learning", layout="wide")

st.title("🤖 Warehouse Robot Q-Learning Optimizer")
st.markdown("Watch the robot learn to navigate the warehouse and pick up items optimally!")

# Simulation Settings
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# 0=Free, 1=Obstacle, 2=Goal
GRID_MAP = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 2],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

START_POS = (0, 0)
GOAL_POS = (2, 4)

# Colors for Matplotlib (0: White, 1: Red/Obstacle, 2: Green/Goal, 3: Blue/Robot)
cmap = ListedColormap(['white', '#e74c3c', '#2ecc71', '#3498db'])

st.sidebar.header("🧠 Q-Learning Hyperparameters")
alpha = st.sidebar.slider("Learning Rate (Alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.1, 1.0, 0.9)
episodes = st.sidebar.slider("Episodes", 100, 1000, 300)

def get_reward(state):
    r, c = state
    if state == GOAL_POS:
        return 100
    elif GRID_MAP[r, c] == 1:
        return -50
    else:
        return -1

def step(state, action_idx):
    r, c = state
    if action_idx == 0: r = max(0, r - 1)     # UP
    elif action_idx == 1: r = min(GRID_SIZE - 1, r + 1) # DOWN
    elif action_idx == 2: c = max(0, c - 1)     # LEFT
    elif action_idx == 3: c = min(GRID_SIZE - 1, c + 1) # RIGHT
    
    new_state = (r, c)
    reward = get_reward(new_state)
    
    # Bounce off obstacles
    if GRID_MAP[r, c] == 1: 
        new_state = state 
        
    done = (new_state == GOAL_POS)
    return new_state, reward, done

def plot_grid(state):
    display_grid = GRID_MAP.copy()
    display_grid[state[0], state[1]] = 3 # Render robot
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(display_grid, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(np.arange(-.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-.5, GRID_SIZE, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

if st.button("Train Robot 🚀"):
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Live stats
    metrics_cols = st.columns(3)
    ep_metric = metrics_cols[0].empty()
    reward_metric = metrics_cols[1].empty()
    steps_metric = metrics_cols[2].empty()

    st.subheader("Live Training View 🎥")
    placeholder = st.empty()
    
    epsilon = 1.0
    epsilon_decay = 0.99
    rewards_history = []
    
    for episode in range(episodes):
        state = START_POS
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100: # Limit steps to prevent infinite loop
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, len(ACTIONS))
            else:
                action = np.argmax(q_table[state[0], state[1]])
                
            new_state, reward, done = step(state, action)
            
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[new_state[0], new_state[1]])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state[0], state[1], action] = new_value
            
            state = new_state
            total_reward += reward
            steps += 1
            
        epsilon = max(0.01, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        # Render a frame in the UI periodically
        if episode % 50 == 0 or episode == episodes - 1:
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            status_text.text(f"Training in progress... Episode {episode+1}/{episodes}")
            ep_metric.metric("Episode", episode + 1)
            reward_metric.metric("Total Reward", total_reward)
            steps_metric.metric("Steps Taken", steps)
            
            fig = plot_grid(state)
            placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.1)
            
    status_text.success("✅ Training Complete!")
    
    # -----------------------------------
    # Final Plots and Validation
    # -----------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Learning Curve")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(rewards_history, color='blue', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title("Reward per Episode")
        st.pyplot(fig2)
        plt.close(fig2)

    with col2:
        st.subheader("✨ Optimal Path Viz")
        path_placeholder = st.empty()
        
        state = START_POS
        done = False
        opt_steps = 0
        
        while not done and opt_steps < 20:
            action = np.argmax(q_table[state[0], state[1]])
            new_state, _, done = step(state, action)
            state = new_state
            opt_steps += 1
            
            fig = plot_grid(state)
            path_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.4) # Slight delay for cool animation
            
        if done:
            st.success("Robot reached the Goal! 🎉")
