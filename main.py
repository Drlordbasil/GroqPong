import numpy as np
import pygame
import json
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
else:
    print("GROQ_API_KEY loaded successfully.")

class ConversationMemory:
    def __init__(self, memory_file="conversation_memory.json"):
        self.memory_file = memory_file
        self.history = self.load_memory()

    def save_context(self, role, content):
        self.history.append({"role": role, "content": content})
        self.save_memory()

    def get_history(self):
        return self.history

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def save_memory(self):
        with open(self.memory_file, "w") as file:
            json.dump(self.history, file, indent=2)

def create_chat_groq(temperature=0, model_name="mixtral-8x7b-32768"):
    return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)

def chat_with_groq(system_prompt, user_prompt, chat_instance=None, memory=None):
    if chat_instance is None:
        chat_instance = create_chat_groq()
    if memory is None:
        memory = ConversationMemory()

    memory.save_context("user", user_prompt)
    history = memory.get_history()
    messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + \
               [HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user" else SystemMessagePromptTemplate.from_template(msg["content"]) for msg in history] + \
               [HumanMessagePromptTemplate.from_template(user_prompt)]
    prompt = ChatPromptTemplate.from_messages(messages)
    response = chat_instance.invoke(prompt.format_prompt())
    memory.save_context("assistant", response.content)
    return response

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1, exploration_decay=0.99):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q

    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

class PongEnv:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.paddle_width = 10
        self.paddle_height = 60
        self.paddle_speed = 5
        self.ball_size = 10
        self.ball_speed_x = 3
        self.ball_speed_y = 3
        self.left_paddle_y = self.height // 2 - self.paddle_height // 2
        self.right_paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2 - self.ball_size // 2
        self.ball_y = self.height // 2 - self.ball_size // 2
        self.left_score = 0
        self.right_score = 0

    def reset(self):
        self.left_paddle_y = self.height // 2 - self.paddle_height // 2
        self.right_paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2 - self.ball_size // 2
        self.ball_y = self.height // 2 - self.ball_size // 2
        self.ball_speed_x = 3
        self.ball_speed_y = 3
        self.left_score = 0
        self.right_score = 0
        return self._get_state()

    def _get_state(self):
        return (self.left_paddle_y, self.right_paddle_y, self.ball_x, self.ball_y, self.ball_speed_x, self.ball_speed_y)

    def step(self, left_action, right_action):
        # Update paddles based on actions
        if left_action == 0:
            self.left_paddle_y -= self.paddle_speed
        elif left_action == 1:
            self.left_paddle_y += self.paddle_speed
        if right_action == 0:
            self.right_paddle_y -= self.paddle_speed
        elif right_action == 1:
            self.right_paddle_y += self.paddle_speed

        # Keep paddles within the screen boundaries
        self.left_paddle_y = max(0, min(self.left_paddle_y, self.height - self.paddle_height))
        self.right_paddle_y = max(0, min(self.right_paddle_y, self.height - self.paddle_height))

        # Update ball position
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # Check for collision with paddles
        if (
            self.ball_x <= self.paddle_width
            and self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height
        ):
            self.ball_speed_x = -self.ball_speed_x * 1.1  # Increase ball speed after collision
        elif (
            self.ball_x >= self.width - self.paddle_width - self.ball_size
            and self.right_paddle_y <= self.ball_y <= self.right_paddle_y + self.paddle_height
        ):
            self.ball_speed_x = -self.ball_speed_x * 1.1  # Increase ball speed after collision

        # Check for collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_speed_y = -self.ball_speed_y

        # Check for scoring
        reward_left = 0
        reward_right = 0
        done = False
        if self.ball_x <= 0:
            self.right_score += 1
            reward_right = 1
            reward_left = -1
            done = True
        elif self.ball_x >= self.width - self.ball_size:
            self.left_score += 1
            reward_left = 1
            reward_right = -1
            done = True

        return self._get_state(), reward_left, reward_right, done

    def render(self, screen):
        screen.fill((0, 0, 0))
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (0, self.left_paddle_y, self.paddle_width, self.paddle_height),
        )
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (self.width - self.paddle_width, self.right_paddle_y, self.paddle_width, self.paddle_height),
        )
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (self.ball_x, self.ball_y, self.ball_size, self.ball_size),
        )
        pygame.display.flip()

def train_agents(env, left_agent, right_agent, num_episodes, render=False):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            left_action = left_agent.choose_action(state)
            right_action = right_agent.choose_action(state)
            next_state, reward_left, reward_right, done = env.step(left_action, right_action)
            left_agent.update_q_table(state, left_action, reward_left, next_state)
            right_agent.update_q_table(state, right_action, reward_right, next_state)
            state = next_state

            if render:
                env.render(screen)
                clock.tick(60)

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        left_agent.decay_exploration_rate()
        right_agent.decay_exploration_rate()

        # Integrate with the chatbot
        system_prompt = "You are a Pong AI assistant. Provide a summary of the game and offer insights on the AI agents' performance."
        user_prompt = f"Episode: {episode+1}, Left Score: {env.left_score}, Right Score: {env.right_score}. How did the AI agents perform in this episode?"
        response = chat_with_groq(system_prompt, user_prompt)
        print(f"Episode: {episode+1}")
        print(f"Chatbot Response: {response.content}")

# Initialize Pygame
pygame.init()

# Create the Pong environment
env = PongEnv()

# Create two instances of QLearningAgent
left_agent = QLearningAgent(num_actions=2)
right_agent = QLearningAgent(num_actions=2)

# Set up the Pygame window
screen = pygame.display.set_mode((env.width, env.height))
clock = pygame.time.Clock()

# Training loop
num_episodes = 1000
train_agents(env, left_agent, right_agent, num_episodes, render=True)

# Close the Pygame window
pygame.quit()
