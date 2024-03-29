import numpy as np
import gym
from openai_test import chat
from gym import spaces
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, FaceVariable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from source_text import GameDescriber, TransitionTranslator, PollutionSourceSampleStorage
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 例如使用微软雅黑
matplotlib.rcParams['font.family']='sans-serif'


# 网格参数
nx, ny = 100, 100
dx, dy = 1.0, 1.0
t_step = 50
num_drones=2
# 创建二维网格
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

class PollutionSourceEnv(gym.Env):
    def __init__(self, grid_size=100, num_drones=2):
        super(PollutionSourceEnv, self).__init__()
        self.grid_size = grid_size
        self.num_drones = num_drones

        # 定义动作空间和观察空间
        self.action_space = spaces.MultiDiscrete([5] * self.num_drones)   # 每架无人机有上下左右移动
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)

        # 初始化无人机状态
        self.drones_positions = None
        self.drones_last_action = np.zeros((num_drones, 1))
        self.drones_highest_concentration = np.zeros(num_drones)
        self.time_since_highest = np.zeros(num_drones)

        # 风速和风向
        self.wind_speed = np.random.uniform(0, 5)  # 随机风速
        self.wind_direction = np.random.uniform(0, 2 * np.pi)  # 随机风向

        # 创建污染物扩散模型
        self.mesh = Grid2D(dx=1.0, dy=1.0, nx=grid_size, ny=grid_size)
        self.concentration = CellVariable(name="污染物浓度", mesh=self.mesh, value=0.)
        self.diffCoeff = FaceVariable(mesh=mesh, value=10)
        # self.diffCoeff = 10  # 扩散系数
        self.concentration_data = np.zeros((t_step, nx, ny))
        # 初始化污染源
        self.set_pollution_source()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
    def set_pollution_source(self):
        source_radius = 5  # 假设所有污染源都有相同的半径
        # 生成每个污染源的位置，可以改变
        self.sources = [(20, 50), (80, 20), (30, 80)]
        # 为每个污染源设置污染浓度
        for source_x, source_y in self.sources:
            self.concentration.setValue(1., where=((self.mesh.x - source_x) ** 2 + (
                        self.mesh.y - source_y) ** 2) < source_radius ** 2)

    def set_buildings(self):
        self.buildings = [(25, 35, 45, 55), (30, 40, 60, 70), (50, 60, 20, 30), (70, 80, 50, 60), (10, 20, 30, 40)]
        for building in self.buildings:
            xmin, xmax, ymin, ymax = building
            self.obstacle = ((mesh.faceCenters[0] >= xmin) & (mesh.faceCenters[0] <= xmax) &
                        (mesh.faceCenters[1] >= ymin) & (mesh.faceCenters[1] <= ymax))
            self.diffCoeff.setValue(0., where=self.obstacle)

    def estimate_source_success(self, drone_pos):
        # 判断无人机是否接近任何一个污染源
        success_radius = 5  # 定义成功半径

        # 假设 self.sources 是一个包含所有污染源位置的列表
        # 每个污染源的位置格式为 (x, y)
        for source_x, source_y in self.sources:
            distance_to_source = np.linalg.norm(np.array([source_x, source_y]) - np.array(drone_pos))
            if distance_to_source <= success_radius:
                return True  # 如果无人机在任何一个污染源的成功半径内，返回 True

        return False  # 如果无人机不在任何一个污染源的成功半径内，返回 False
    def reset(self):
        # 重置污染物浓度
        self.concentration.setValue(0.)
        # 重新设置污染源
        self.set_pollution_source()
        self.set_buildings()
        # 重置无人机位置
        # self.drones_positions = np.random.randint(0, self.grid_size, (self.num_drones, 2))
        self.drones_positions = np.array([[20,70],[70,20]])
        self.drones_last_action = np.full(self.num_drones,4)  # 确保这是一个一维数组
        self.drones_highest_concentration = np.zeros(self.num_drones)
        self.time_since_highest = np.zeros(self.num_drones)
        # 构建初始观察值
        obs = []
        for drone_pos in self.drones_positions:
            local_concentration = self.get_concentration_at(drone_pos)
            obs.append([float(local_concentration),
                        0.0,  # 初始最高浓度设置为0
                        0.0,  # 初始时间设置为0
                        0,  # 初始动作设置为0（原地不动）
                        float(drone_pos[0]),
                        float(drone_pos[1])])


        # 计算奖励
        rewards = []
        for i, drone_pos in enumerate(self.drones_positions):
            current_concentration = self.get_concentration_at(drone_pos)
            reward = -1  # 默认奖励为-1

            if current_concentration > self.drones_highest_concentration[i]:
                self.drones_highest_concentration[i] = current_concentration
                reward = 0.1  # 更新最高浓度奖励为0.1

            # 估计成功的逻辑（需要具体实现）
            if self.estimate_source_success(drone_pos):
                reward = 100

            rewards.append(reward)

        term=np.zeros(self.num_drones, dtype=bool)
        for i, drone_pos in enumerate(self.drones_positions):
            if self.estimate_source_success(drone_pos):
                term[i]=True
        # 返回初始观察值
        return np.array(obs),rewards, term,{}

    def step(self, actions):
        # 更新无人机位置
        for i in range(self.num_drones):
            action = actions[i]
            dx, dy = 0, 0
            if action == 0: dy = 1   # 向上
            elif action == 1: dy = -1  # 向下
            elif action == 2: dx = -1  # 向左
            elif action == 3: dx = 1   # 向右
            # action == 4 是原地不动，dx和dy保持为0

        # 根据动作更新无人机位置
        #     self.drones_positions[i, 0] = np.clip(self.drones_positions[i, 0] + dx, 0, self.grid_size - 1)
        #     self.drones_positions[i, 1] = np.clip(self.drones_positions[i, 1] + dy, 0, self.grid_size - 1)
            new_x = self.drones_positions[i, 0] + dx
            new_y = self.drones_positions[i, 1] + dy
            collision = False
            for building in self.buildings:
                xmin, xmax, ymin, ymax = building
                if (new_x>=xmin)&(new_y>=ymin)&(new_x<=xmax)&(new_y<=ymax):
                    collision = True
                    break

            # 如果没有碰撞，则更新位置
            if not collision:
                self.drones_positions[i, 0] = np.clip(new_x, 0, self.grid_size - 1)
                self.drones_positions[i, 1] = np.clip(new_y, 0, self.grid_size - 1)

            self.time_since_highest[i] = self.time_since_highest[i] + 1
        # 更新污染物浓度
        eq = TransientTerm() == DiffusionTerm(coeff=self.diffCoeff)
        eq.solve(var=self.concentration, dt=0.1)


        # 计算奖励
        rewards = []
        for i, drone_pos in enumerate(self.drones_positions):
            current_concentration = self.get_concentration_at(drone_pos)
            reward = -1  # 默认奖励为-1

            if current_concentration > self.drones_highest_concentration[i]:
                self.drones_highest_concentration[i] = current_concentration
                self.time_since_highest[i]=0
                reward = 0.1  # 更新最高浓度奖励为0.1

            # 估计成功的逻辑（需要具体实现）
            if self.estimate_source_success(drone_pos):
                reward = 100

            rewards.append(reward)

        # 更新上一个动作
        self.drones_last_action = actions.copy()

        # 更新观察
        obs = []
        for i, drone_pos in enumerate(self.drones_positions):
            local_concentration = self.get_concentration_at(drone_pos)
            obs.append([float(local_concentration),
                        float(self.drones_highest_concentration[i]),
                        float(self.time_since_highest[i]),
                        float(self.drones_last_action[i]),  # 确保这是一个单个的数值
                        float(drone_pos[0]),
                        float(drone_pos[1])])


        term=np.zeros(self.num_drones, dtype=bool)
        for i, drone_pos in enumerate(self.drones_positions):
            if self.estimate_source_success(drone_pos):
                term[i]=True


        # ...之后的代码...
        return np.array(obs), np.array(rewards), term, {}

    def get_concentration_at(self, position):
        # 将一维数组转换为二维数组
        concentration_2d = self.concentration.value.reshape((self.grid_size, self.grid_size))
        x, y = int(position[0]), int(position[1])
        return concentration_2d[x, y]

translator = TransitionTranslator()
storage = PollutionSourceSampleStorage()
describer = GameDescriber()
print(describer.describe_game())
print(describer.describe_action())


env = PollutionSourceEnv()
observation, reward,terminated,info = env.reset()
storage.add_sample(observation, [4,4], observation, reward, terminated, info)
def update(frame):
    # action = env.action_space.sample()
    infos = storage.get_samples()
    word = translator.translate(infos)
    action = chat(word)
    new_observation, reward, terminated, info = env.step(action)
    global observation
    storage.add_sample(observation, action, new_observation, reward, terminated, info)

    print(word)
    for i in range(num_drones):
        if terminated[i]:
            # observation, info, _, _ = env.reset()
            print(f"Drone{i} Find terminated!")
            # env.close()
    env.concentration_data[frame, :, :] = env.concentration.value.reshape((nx, ny))
    env.ax.clear()
    env.ax.imshow(env.concentration_data[frame, :, :], origin='lower', extent=(0, nx * dx, 0, ny * dy))
    env.ax.set_title(f"时间步 {frame}")
    for building in env.buildings:
        xmin, xmax, ymin, ymax = building
        env.ax.fill_betweenx([ymin, ymax], xmin, xmax, color='grey', alpha=0.5)
    for j in range(env.num_drones):
        env.ax.scatter(env.drones_positions[j, 0], env.drones_positions[j, 1], c='red', marker='o')  # 使用红色圆点表示无人机

ani = FuncAnimation(env.fig, update, frames=t_step, repeat=False)

np.save('concentration_data.npy', env.concentration_data)
plt.show()

# Close the environment
env.close()

# # Retrieve stored transitions
# infos = storage.get_samples()
#
#
# # Translate the transitions to text
# words = translator.translate(infos)
#
#
# for description in words:
#     print(description)