from machin.utils.helper_classes import Switch
from machin.frame.noise.param_space_noise import perturb_model
from machin.utils.visualize import visualize_graph
import torch as t

dims = 5

t.manual_seed(0)
model = t.nn.Linear(dims, dims)
optim = t.optim.Adam(model.parameters(), 1e-3)
p_switch, r_switch = Switch(), Switch()
rst_func = perturb_model(model, p_switch, r_switch)
r_switch.on()

# turn off/on the perturbation switch to see the difference
p_switch.on()

# do some sampling
action = model(t.ones([dims]))

# Visualize will not show any leaf noise tensors
# because they are created in t.no_grad() context.
visualize_graph(action, exit_after_vis=False)

# do some training
loss = (action - t.ones([dims])).sum()
loss.backward()
rst_func()
optim.step()
print(model.weight)
