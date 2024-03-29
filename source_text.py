class ObsTranslator:
    def __init__(self):
        pass
    def translate(self, states):
        # 为每个智能体生成描述，并包括编号
        descriptions = [
            f"Drone {i+1}: Located at coordinates ({state[4]}, {state[5]}), "
            f"with a local pollution concentration of {state[0]:.2f}. "
            f"Highest concentration detected: {state[1]:.2f}, "
            f"timesteps since highest: {state[2]}. "
            f"last action: {'stay in place' if int(state[3]) == 4 else ['move up', 'move down', 'move left', 'move right'][int(state[3])]}."
            for i, state in enumerate(states[0])
        ]

        # 使用换行符连接各个无人机的描述
        return "\n".join(descriptions)

    # def translate(self, states):
    #     descriptions = []
     
    #     for state in states[0]:
      
    #         drone_position_x, drone_position_y = state[4], state[5]
    #         concentration = state[0]
    #         highest_concentration = state[1]
    #         time_since_highest = state[2]
    #         last_action = int(state[3])

    #         position_desc = (f"A drone is located at coordinates ({drone_position_x}, {drone_position_y}), "
    #                          f"with a local pollution concentration of {concentration:.2f}.")
    #         concentration_desc = (f"The highest concentration detected so far is {highest_concentration:.2f}, "
    #                               f"which was {time_since_highest} timesteps ago.")
    #         action_desc = f"The last action taken was {'stay in place' if last_action == 4 else ['move up', 'move down', 'move left', 'move right'][last_action]}."

    #         res = f"{position_desc} {concentration_desc} {action_desc}"
    #         descriptions.append(res)
    #     return descriptions



class GameDescriber:
    def __init__(self):
        pass

    def describe_goal(self):
        return "The goal is to locate the source of pollution by detecting the highest concentration point."

    def describe_game(self):
        return ("In the Pollution Source Tracking game, you control multiple drones to locate a pollution source in a two-dimensional grid. "
                "Each drone can move up, down, left, right, or stay in place. The environment is affected by wind, influencing the pollution dispersion. "
                "The game ends when a drone locates the pollution source or after a predefined number of timesteps.")

    def describe_action(self):
        return ("Your Next Move: \n Please choose an action for each drone. Type '0' to move up, '1' to move down, '2' to move left, "
                "'3' to move right, or '4' to stay in place. Ensure you only provide action numbers from the valid action list, i.e., [0, 1, 2, 3, 4].")


# class TransitionTranslator(ObsTranslator):
#     def translate(self, infos, is_current=False):
#         descriptions = []
#         if is_current:
#             states_desc = ObsTranslator().translate([info['state'] for info in infos])
#             return "Current State: " + " | ".join(states_desc)
#
#         for info in infos:
#             assert 'state' in info, "info should contain state information"
#             state_desc = ObsTranslator().translate([info['state']])
#
#             # 动作描述列表
#             action_mapping = ['move up', 'move down', 'move left', 'move right', 'stay in place']
#
#             # 使用列表推导和格式化字符串生成紧凑的动作描述
#             action_desc = ", ".join(f"Drone {i+1} Action: {action_mapping[action]} ({action})"
#                                                 for i, action in enumerate(info['action']))
#
#             reward_desc = f"Result: Reward of {info['reward']}, "
#             next_state_desc = ObsTranslator().translate([info['next_state']])
#
#
#             # descriptions.append(f"{state_desc}.\n {action_desc} \n {reward_desc} \n Next State: {next_state_desc}")
#             descriptions.append(f"{next_state_desc}\n")
#
#         return descriptions


#new version
class TransitionTranslator(ObsTranslator):
    def translate(self, info, is_current=False):
        descriptions = ""
        assert 'state' in info, "info should contain state information"
        state_desc = ObsTranslator().translate([info['state']])

        # 动作描述列表
        action_mapping = ['move up', 'move down', 'move left', 'move right', 'stay in place']

        # 使用列表推导和格式化字符串生成紧凑的动作描述
        action_desc = ", ".join(f"Drone {i + 1} Action: {action_mapping[action]} ({action})"
                                for i, action in enumerate(info['action']))

        reward_desc = f"Result: Reward of {info['reward']}, "
        next_state_desc = ObsTranslator().translate([info['next_state']])

        # descriptions.append(f"{state_desc}.\n {action_desc} \n {reward_desc} \n Next State: {next_state_desc}")
        descriptions=f"{next_state_desc}\n"

        return descriptions








class PollutionSourceSampleStorage:
    def __init__(self):
        self.samples = []

    def add_sample(self, observation, action, next_state, reward, terminated, info):
        """
        添加一个样本到存储中。

        :param observation: 环境中的观察。
        :param action: 执行的动作。
        :param next_state: 执行动作后的下一个状态。
        :param reward: 由于执行动作获得的奖励。
        :param terminated: 是否终止。
        :param truncated: 是否截断。
        :param info: 额外的信息。
        """


        sample = {
            'state': observation,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'terminated': terminated,      
            'info': info
        }
        self.samples.append(sample)


    def get_samples(self):
        """
        获取存储的所有样本。

        :return: 包含所有样本的列表。
        """
        return self.samples.pop(0)
