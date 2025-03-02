from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = """
## Task

Explain the Decision-Making Process Based on Visual Embeddings and Decision Tree You are an AI assistant designed to explain decision-making processes in a reinforcement learning system that utilizes visual embeddings and decision trees. Your goal is to generate a natural language explanation of how an input image leads to a specific decision, breaking down the importance of objects in the scene at each decision node. 

## Input Data

- The decision tree follows a sequential decision-making process based on embedding values at different dimensions.
- Each decision node checks a specific dimension (e.g., `x_38 = 1.22 <= -1.42`), determining whether to follow the left or right path.
- Objects within the image influence these decisions, with corresponding importance scores (weights) indicating how much they contribute to the decision.
- The final decision assigns a skill (e.g., `Skill 0`). 

## Example Decision Path

x_38 = -2.546 <= -1.424 -> x_125 = -0.472 <= 0.631 -> x_75 = 0.001 > -0.253 -> Skill 0

## Example Object Influence 

x_38
Object 0: The light switch (importance: 0.0290)
Object 1: A robotic arm (importance: 0.0339)
Object 2: Microwave oven (importance: 0.0423)
x_125
Object 0 (microwave): 0.0539
Object 1 (slide cabinet): 0.0796
Object 2 (hinge cabinet): 0.0552

## Output Instructions

- Generate a structured, easy-to-understand explanation of the decision process.
- Describe the **key objects** that contributed to each decision.
- Highlight the **most influential objects** at each step.
- Use **causal language** (e.g., "Because the stove dials were detected, the decision tree favored this path..."). 
- Conclude by summarizing why the final skill was selected.

Now, the the decision path is {decision_path}. 

For each decision node, the corresponding importance score is:
{score}
"""

prompt = PromptTemplate.from_template(template)

class FeastPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        return prompt.format(**kwargs)

prompt_template = FeastPromptTemplate(input_variables=["decision_path", "score"])

model = ChatOpenAI(
    api_key="sk-KH82aab3b53176ce707cad1961fcc9491c39d4279cdad5zo",
    base_url="https://api.gptsapi.net/v1",
    model="gpt-4o",
)

parser = StrOutputParser()
chain = model | parser