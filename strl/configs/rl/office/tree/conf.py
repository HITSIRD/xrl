from strl.configs.rl.office.base_conf import *
from strl.rl.policies.tree_policies import CARTPolicy
from strl.rl.agents.tree_agent import CARTAgent

max_depth = 12

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    policy=CARTPolicy,
))

configuration.update(AttrDict(
    agent=CARTAgent
))

policy_params.update(AttrDict(
    is_index=False,
    type='regressor',
    max_depth=max_depth,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         f"/home/wenyongyan/文档/CART/action/office/cart_100_d{max_depth}.pkl"),
))
