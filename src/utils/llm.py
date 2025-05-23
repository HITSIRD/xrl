from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

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

template = """
你是一名 AI 助手，负责解释在强化学习系统中基于视觉输入的决策过程**。你的**目标是生成自然语言解释**，说明输入图像如何导致特定决策，并分析场景中不同物体对决策结果重要性。

## **输入数据**

- 场景中的物体会影响决策结果，并具有**重要性权重**，表示它们对决策的贡献程度。
- 最终决策会输出一个**技能（skill）**（例如 Skill 0）。

## **示例物体影响**

Object 0 (microwave): 0.0539
Object 1 (slide cabinet): 0.0796
Object 2 (hinge cabinet): 0.0552

## **输出说明**

- 生成结构清晰、易于理解且语言简短的决策过程解释，不要过于详细的解释。
- 强调最具影响力的物体，突出它们在决策中的作用。
- 使用因果语言。
- 不要使用markdown格式，也不能用**表示粗体，使用普通文本来描述，只用一段句子，不要分段描述。
- 最后总结为何选择最终技能（Skill）。
- 用最简短的语言来描述，只关注重要物体即可，省略次要内容。

现在，决策结果为技能{skill_index}，意图完成{current_task}任务。

不同物体的重要性权重为：
{score}
"""

prompt = PromptTemplate.from_template(template)

class FeastPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        return prompt.format(**kwargs)

# prompt_template = FeastPromptTemplate(input_variables=["decision_path", "score"])
prompt_template = FeastPromptTemplate(input_variables=["skill_index", "current_task", "score"])

model = ChatOpenAI(
    api_key="sk-KH82aab3b53176ce707cad1961fcc9491c39d4279cdad5zo",
    base_url="https://api.gptsapi.net/v1",
    model="gpt-4o",
)

parser = StrOutputParser()
chain = model | parser