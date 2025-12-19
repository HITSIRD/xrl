import base64
import io

from PIL import Image
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    api_key="sk-KH82aab3b53176ce707cad1961fcc9491c39d4279cdad5zo",
    base_url="https://api.gptsapi.net/v1",
    model="gpt-4o",
)

image_prompt = """
根据该图像描述这个场景中的物体以及状态，把这个场景转成一个适合做机器人任务标注的structured JSON描述，包含物体类别（class）、位置（粗略的相对区域）、状态（是否打开、是否包含内容、是否被操作等）示例{
  "scene": "kitchen_table_with_robot",
  "objects": [
    {
      "class": "robot_arm",
      "position": "left",
      "state": "active",
      "interaction": "gripper facing microwave, possibly retrieving bread"
    },
    {
      "class": "microwave",
      "position": "back_center_top",
      "state": "door_open",
      "contains": ["bread"]
    }
  ]
}

只输出json，不要输出其他任何内容。
"""


def numpy_to_base64(image_np):
    pil_image = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_image_message(image_np, detail="auto"):
    base64_image = numpy_to_base64(image_np)
    return HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": detail}
            },
            {"type": "text",
             "text": image_prompt}
        ]
    )


# template = """
# ## Task
#
# Explain the Decision-Making Process Based on Visual Embeddings and Decision Tree. You are an AI assistant designed to explain decision-making processes in a reinforcement learning system that utilizes visual embeddings and decision trees. Your goal is to generate a natural language explanation of how an input image leads to a specific decision, breaking down the importance of objects in the scene at each decision node.
#
# ## Input Data
#
# - The decision tree follows a sequential decision-making process based on embedding values at different dimensions.
# - Each decision node checks a specific dimension (e.g., `x_38 = 1.22 <= -1.42`), determining whether to follow the left or right path.
# - Objects within the image influence these decisions, with corresponding importance scores (weights) indicating how much they contribute to the decision.
# - The final decision assigns a skill (e.g., `Skill 0`).
#
# ## Example Decision Path
#
# x_38 = -2.546 <= -1.424 -> x_125 = -0.472 <= 0.631 -> x_75 = 0.001 > -0.253 -> Skill 0
#
# ## Example Object Influence
#
# x_38
# Object 0: The light switch (importance: 0.0290)
# Object 1: A robotic arm (importance: 0.0339)
# Object 2: Microwave oven (importance: 0.0423)
# x_125
# Object 0 (microwave): 0.0539
# Object 1 (slide cabinet): 0.0796
# Object 2 (hinge cabinet): 0.0552
#
# ## Output Instructions
#
# - Generate a structured, easy-to-understand explanation of the decision process.
# - Describe the **key objects** that contributed to each decision.
# - Highlight the **most influential objects** at each step.
# - Use **causal language** (e.g., "Because the stove dials were detected, the decision tree favored this path...").
# - Conclude by summarizing why the final skill was selected.
#
# Now, the the decision path is {decision_path}.
#
# For each decision node, the corresponding importance score is:
# {score}
# """

# template = """
# 你是一名 AI 助手，负责解释在强化学习系统中基于视觉输入的决策过程**。你的**目标是生成自然语言解释**，说明输入图像如何导致特定决策，并分析场景中不同物体对决策结果重要性。
#
# ## **输入数据**
#
# - 场景中的物体会影响决策结果，并具有**重要性权重**，表示它们对决策的贡献程度。
# - 最终决策会输出一个**技能（skill）**（例如 Skill 0）。
#
# ## **示例物体影响**
#
# Object 0 (microwave): 0.0539
# Object 1 (slide cabinet): 0.0796
# Object 2 (hinge cabinet): 0.0552
#
# ## **输出说明**
#
# - 生成结构清晰、易于理解且语言简短的决策过程解释，不要过于详细的解释。
# - 强调最具影响力的物体，突出它们在决策中的作用。
# - 使用因果语言。
# - 不要使用markdown格式，也不能用**表示粗体，使用普通文本来描述，只用一段句子，不要分段描述。
# - 最后总结为何选择最终技能（Skill）。
# - 用最简短的语言来描述，只关注重要物体即可，省略次要内容。
#
# 现在，决策结果为技能{skill_index}，意图完成{current_task}任务。
#
# 不同物体的重要性权重为：
# {score}
# """

# - 微波炉
# - 滑轨壁橱
# - 合页壁橱
# - 上炉灶开关
# - 下炉灶开关
# - 水壶

# 场景物体：
# - bottom burner switch
# - top burner switch
# - light switch
# - slide cabinet
# - hinge cabinet
# - microwave
# - kettle
#
# 技能与子任务映射：
# - 技能 0：打开微波炉
# - 技能 1：打开滑轨壁橱
# - 技能 2：打开合页壁橱
# - 技能 3：开灯
# - 技能 4：打开上炉灶开关
# - 技能 5：打开下炉灶开关
# - 技能 6：移动水壶

# 不同技能选择概率：
# - 技能 0：0.95
# - 技能 1：0.01
# - 技能 2：0.01
# - 技能 3：0.00
# - 技能 4：0.00
# - 技能 5：0.00
# - 技能 6：0.03

template = """
你是一个智能解释器，负责解释一个机械臂的行为。机械臂在一个厨房的视觉环境中执行任务，观测是RGB图像。机械臂需要根据观测从多个技能中选择一个并执行。
当前的任务是完成厨房收纳，需要将水果依次放入冰箱中，然后将零食依次放入储藏柜中。

现在你获得以下信息：
1. 场景描述，用json格式表示；
2. 场景中不同物体列表，按照可解释方法的计算出的每个物体的显著性分数从高到低排序，显著性分数表示该物体对当前决策的重要性；
3. 当前选择的技能以及历史选择的技能列表。

你的要求是：
- 根据场景描述、技能索引和显著物体列表，解释机器人当前为什么执行该技能；
- 用简短的语言给出你的推理过程，要做复杂分析，让人看一眼就能理解；
- 注意你的解释对象是一个对机器学习背景缺乏了解的普通人，因此不要出现不要出现类似“显著性”这样的专业术语，要通俗易懂。
- 只给出简要推理过程，不需要给出额外信息。
- 用中文描述，不要出现英文。

场景描述：
{scene_description}

输入示例：

历史选择技能序列：
['打开冰箱']

物体显著性排序：
['orange', 'lemon', 'cheezit', 'mango', 'jello']

示例输出格式：

推理过程：XXXX

------

现在，历史选择技能为{history}，并且不同物体的显著性由高到低排序为{sorted_objects}
解释：
"""

# template = """
# 你是一个智能解释器，负责解释一个机械臂的行为。机械臂在一个厨房的视觉环境中执行任务，机械臂的观测是RGB图像。机械臂需要根据观测从多个技能中选择一个并执行。
#
# 当前的任务是加热面包，需要将面包放入微波炉中加热，然后放到盘上。
#
# 现在你获得以下信息：
# 1. 场景描述，用json格式表示；
# 2. 可解释方法计算出的每个物体的显著性分数，表示该物体对机器人决策的重要性；
# 3. 每个技能与其对应子任务之间的映射关系。
#
# 你的要求是：
# - 根据场景描述、技能索引和显著性列表，解释机器人当前为什么执行该技能；
# - 用简短的语言给出你的推理过程；
# - 注意显著性分数的计算结果是事后解释和分析，不一定和实际选择的技能相符，对于这种情况要指出来。
# - 注意你的解释对象是一个对机器学习背景缺乏了解的普通人，因此不要出现原始的显著性分数结果，也不要出现类似“显著性”这样的专业术语，要通俗易懂。
# - 用中文描述。
#
# 场景描述：
# {scene_description}
#
# 历史选择技能序列：
# [0]
#
# 执行技能：3
#
# 技能与子任务映射：
# - 技能 0：打开微波炉
# - 技能 1：关闭微波炉
# - 技能 2：设置时间
# - 技能 3：面包放入微波炉
# - 技能 4：面包放入盘中
# - 技能 5：结束
#
# 输入示例：
#
# 显著性分数：
# - Object 0 (microwave): 17.12
# - Object 1 (bread): 72.85
# - Object 2 (switch): -3.66
#
# 示例输出格式：
#
# 当前技能：面包放入微波炉
# 推理过程：XXX
#
# ---
#
# 当前，机械臂选择了技能索引{skill_index}，历史选择技能为{history}, 并且不同物体的重要性权重为
# {score}
# 解释：
# """

prompt = PromptTemplate.from_template(template)


class FeastPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        return prompt.format(**kwargs)


# prompt_template = FeastPromptTemplate(input_variables=["decision_path", "score"])
prompt_template = FeastPromptTemplate(input_variables=["scene_description", "skill", "sorted_objects", "history"])

parser = StrOutputParser()
chain = model | parser

def get_scene_description(img):
    return model.invoke([create_image_message(img)]).content
