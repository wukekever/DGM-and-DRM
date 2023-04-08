import torch


def train_step(
    sol, equation, trainloader, regularizers, optimizer, scheduler, **kwargs
):

    trainloader_interior = trainloader[0]

    optimizer.zero_grad()

    eqn_residual = equation.residual(sol=sol, inputs=trainloader_interior)

    res_eqn_u = eqn_residual["equation"]

    res_boundary = equation.bc(sol=sol)
    res_u_l = res_boundary["bc_left"]
    res_u_r = res_boundary["bc_right"]

    res_eqn = torch.mean(res_eqn_u**2)

    res_bc = torch.mean(res_u_l**2) + torch.mean(res_u_r**2)

    loss = regularizers[0] * res_eqn + regularizers[1] * res_bc

    risk = {}
    risk.update({"total_loss": loss.item()})
    risk.update({"eqn": res_eqn.item()})
    risk.update({"bc": res_bc.item()})

    error = equation.val(sol=sol)

    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()  # clear memory

    return risk, error
