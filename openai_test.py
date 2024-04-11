import requests
import re
url = "https://openai.api2d.net/v1/chat/completions"

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer fk221464-osA3eQTcM0k3udg2csGZkARnqaYCivZc' # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
}
chat_counter = 0  # 跟踪chat函数调用次数
dialogue_history = []
memory = []
file_name = 'text.txt'
with open(file_name, 'r', encoding='utf-8') as file:
     base_text = file.read()
   #conbined_text = base_text + "  " + input
     dialogue_history.append({"role": "system", "content": base_text})
def reflect():
    global dialogue_history
    global memory
    # 假设"反思"只是发送当前的对话历史并获取模型的总结或思考
    reflection_prompt = "Reflect on the drone's movement pattern and based on the aforementioned drone flight trajectory, propose some suggestions for improvement."
    memory.append(reflection_prompt)
    data = {
    "model": "gpt-3.5-turbo",
    "messages":dialogue_history
  }

    response = requests.post(url, headers=headers, json=data)
    reflection_response = response.json().get('choices')[0].get('message').get('content').strip()

    # 将反思结果加入对话历史
    memory.append({"role": "system", "content": reflection_response})
    
    # continue_message = "Next, I will provide the current locations of Drone1 and Drone2, the concentration of pollution at those locations, the highest concentration detected previously, and the time since that highest concentration.You can guide the next movement of the two drones based on this information. Please remember there are three pollution sources instead of one. Please choose an action for each drone. Type '0' to move up, '1' to move down, '2' to move left, '3' to move right, or '4' to stay in place. Ensure you only provide action numbers from the valid action list.More importantly, the output of your content must be a two-dimensional array. For example, if you want Drone1 to move up and Drone2 to move down, you would output: [0, 1].You may include your thought process, but it is essential to ensure that the format of the very end of the output is:So, the output for the next actions of the drones would be: [action 1, action 2].Don't add anything else after this"
    # dialogue_history.append({"role": "system", "content": continue_message})
def chat(input):
  global chat_counter
  global dialogue_history

  # chat_counter += 1  # 每次调用时计数器加1
  # if chat_counter % 5 == 0:
  #     reflect()
  dialogue_history.append({"role": "user", "content": input})
  data = {
    "model": "gpt-3.5-turbo",
    #"messages": [{"role": "user", "content": conbined_text}]
    "messages":dialogue_history
  }

  response = requests.post(url, headers=headers, json=data)
  chatmessage=response.json().get('choices')[0].get('message').get('content').strip()
  dialogue_history.append({"role": "system", "content": chatmessage})

  print(chatmessage)
  # result=[int(chatmessage[-6]),int(chatmessage[-3])]
  cor = re.search('\[(-?\d+)(, ?)(-?\d+)\]', chatmessage)
  try:
    x = int(cor.group(1))
    y = int(cor.group(3))
    result = [x, y]
    # 返回结果
    return result
  except:
      print('chat error')
      return[-1,-1]

  # return chatmessage
  # print("Status Code", response.status_code)
  # return response.json().get('choices')[0].get('message').get('content').strip()

##测试
# input1="""
# Drone 1: Located at coordinates (20.0, 80.0), with a local pollution concentration of 1.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# Drone 2: Located at coordinates (80.0, 20.0), with a local pollution concentration of 0.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# """
# print(chat(input1))
# input2="""
# Drone 1: Located at coordinates (21.0, 80.0), with a local pollution concentration of 1.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# Drone 2: Located at coordinates (81.0, 20.0), with a local pollution concentration of 0.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# """
# print(chat(input2))
# input3="""
# Drone 1: Located at coordinates (22.0, 80.0), with a local pollution concentration of 1.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# Drone 2: Located at coordinates (82.0, 20.0), with a local pollution concentration of 0.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# """
# print(chat(input3))
# input4="""
# Drone 1: Located at coordinates (23.0, 80.0), with a local pollution concentration of 1.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# Drone 2: Located at coordinates (83.0, 20.0), with a local pollution concentration of 0.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# """
# print(chat(input4))
# input5="""
# Drone 1: Located at coordinates (24.0, 80.0), with a local pollution concentration of 1.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# Drone 2: Located at coordinates (84.0, 20.0), with a local pollution concentration of 0.00. Highest concentration detected: 0.00, timesteps since highest: 0.0. Last action: move up.
# """
# print(chat(input5))
# input1="""what is the weather like"""
# input2="""what i said just now"""
# print(chat(input1))
# print(chat(input2))
#改一下