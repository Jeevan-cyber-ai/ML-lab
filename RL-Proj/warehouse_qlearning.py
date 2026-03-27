import numpy as np
import pygame
import matplotlib.pyplot as plt
import random

# --- 1. ENVIRONMENT SETTINGS ---
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

# Colors for Visualization
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (46, 204, 113)   # Goal
RED = (231, 76, 60)      # Obstacle
BLUE = (52, 152, 219)    # Robot
GRAY = (149, 165, 166)   # Grid lines

# Environment Map
# 0 = Free, 1 = Obstacle, 2 = Goal
GRID_MAP = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 2],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

START_POS = (0, 0)
GOAL_POS = (2, 4)

# --- 2. Q-LEARNING PARAMETERS ---
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = len(ACTIONS)

ALPHA = 0.1          # Learning Rate
GAMMA = 0.9          # Discount Factor
EPSILON = 1.0        # Exploration Rate
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01

EPISODES = 300

# Initialize Q-Table (State: row, col -> Action: UP, DOWN, LEFT, RIGHT)
q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# --- 3. HELPER FUNCTIONS ---
def get_reward(state):
    r, c = state
    if state == GOAL_POS:
        return 100
    elif GRID_MAP[r][c] == 1:
        return -50
    else:
        return -1  # Step penalty to force taking the shortest path

def step(state, action_idx):
    r, c = state
    
    if action_idx == 0:   # UP
        r = max(0, r - 1)
    elif action_idx == 1: # DOWN
        r = min(GRID_SIZE - 1, r + 1)
    elif action_idx == 2: # LEFT
        c = max(0, c - 1)
    elif action_idx == 3: # RIGHT
        c = min(GRID_SIZE - 1, c + 1)
    
    new_state = (r, c)
    reward = get_reward(new_state)
    
    # If hitting an obstacle, bounce back to the previous state
    if GRID_MAP[r][c] == 1:
        new_state = state 
        
    done = (new_state == GOAL_POS)
    return new_state, reward, done

def draw_grid(screen, robot_state):
    screen.fill(WHITE)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # Obstacles
            if GRID_MAP[r][c] == 1:
                pygame.draw.rect(screen, RED, rect)
            # Goal
            elif GRID_MAP[r][c] == 2:
                pygame.draw.rect(screen, GREEN, rect)
            
            # Grid borders
            pygame.draw.rect(screen, GRAY, rect, 1)
            
    # Draw Robot
    robot_r, robot_c = robot_state
    # Circle for robot
    pygame.draw.circle(screen, BLUE, (int(robot_c * CELL_SIZE + CELL_SIZE/2), int(robot_r * CELL_SIZE + CELL_SIZE/2)), int(CELL_SIZE/3))
    
    pygame.display.flip()

# --- 4. MAIN LOOP ---
def main():
    global EPSILON
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Warehouse Robot")
    clock = pygame.time.Clock()
    
    rewards_per_episode = []
    steps_per_episode = []
    
    print("🚀 Starting Training...")
    
    for episode in range(EPISODES):
        state = START_POS
        total_reward = 0
        steps = 0
        done = False
        
        # Every 50 episodes, we slow down to visually see the learning progress
        render_this_episode = (episode % 50 == 0) or (episode == EPISODES - 1)
        
        if render_this_episode:
            print(f"🎬 Episode {episode} - Visualizing (Epsilon: {EPSILON:.2f})")
            
        while not done:
            # Handle Pygame quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Epsilon-Greedy Action Selection
            if random.uniform(0, 1) < EPSILON:
                action_idx = random.randint(0, NUM_ACTIONS - 1) # Explore
            else:
                action_idx = np.argmax(q_table[state[0], state[1]]) # Exploit
                
            # Take Step
            new_state, reward, done = step(state, action_idx)
            
            # Update Q-Table
            old_value = q_table[state[0], state[1], action_idx]
            next_max = np.max(q_table[new_state[0], new_state[1]])
            
            new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            q_table[state[0], state[1], action_idx] = new_value
            
            state = new_state
            total_reward += reward
            steps += 1
            
            # Render if needed
            if render_this_episode:
                draw_grid(screen, state)
                time_delay = 100 if episode == EPISODES - 1 else 20  # slower at final episode
                pygame.time.delay(time_delay)
                
        # Decay Epsilon
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        
    print("✅ Training Complete!")
    pygame.quit()
    
    # --- 5. PLOT RESULTS ---
    print("📈 Plotting Reward & Steps Graph...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward Graph
    ax1.plot(rewards_per_episode, color='blue')
    ax1.set_title("Total Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    
    # Steps Graph
    ax2.plot(steps_per_episode, color='red')
    ax2.set_title("Steps Taken per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()

if __name__ == "__main__":
    main()
