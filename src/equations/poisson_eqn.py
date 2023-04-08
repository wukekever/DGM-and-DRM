from math import pi

import numpy as np
import torch


class Poisson(object):
    def __init__(self, config, sol, name="Poisson_Eqn", **kwargs):

        self.alpha = config["physical_config"]["alpha"]

        # device setting
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
        )

        # domain
        self.xmin = config["physical_config"]["x_range"][0]
        self.xmax = config["physical_config"]["x_range"][1]

        # ref
        self.ref_x = torch.linspace(self.xmin, self.xmax, 100)[:, None].to(self.device)

    # inputs: x
    def residual(self, sol, inputs):

        x = inputs
        values, derivatives = self.value_and_grad(sol, x)
        u = values["u"]
        laplace_u = derivatives["space"]

        eqn_res = {}
        f = self.source(inputs=x)
        # residual for Poisson equation
        res_eqn = - laplace_u + torch.pi**2 * u - f
        eqn_res.update({"equation": res_eqn})
        return eqn_res

    def value_and_grad(self, sol, x):
        x.requires_grad = True
        model = sol
        values = {}
        u = self.construct_sol(sol=model, inputs=x)
        values.update({"u": u})

        derivatives = {}
        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones(u.shape).to(self.device),
            create_graph=True,
        )[0]

        laplace_u = torch.autograd.grad(
            outputs=du_dx,
            inputs=x,
            grad_outputs=torch.ones(du_dx.shape).to(self.device),
            create_graph=True,
        )[0]

        derivatives.update({"space": laplace_u})

        return values, derivatives

    # inputs = x
    def construct_sol(self, sol, inputs):
        x = inputs
        sol_fn = sol(x)
        return sol_fn

    # inputs = x
    def source(self, inputs):
        x = inputs
        f = 2.0 * torch.pi**2 * torch.cos(torch.pi * x)
        return f
    
    # # dirichelet_bc
    # def bc(self, sol):
    #     xbc = torch.ones((1, 1)).to(self.device)
    #     xbc_l, xbc_r = (
    #         torch.ones_like(xbc) * self.xmin,
    #         torch.ones_like(xbc) * self.xmax,
    #     )

    #     model = sol
    #     # Left
    #     ubc_l = self.construct_sol(sol=model, inputs=xbc_l)
    #     g_l = 1

    #     # Right
    #     ubc_r = self.construct_sol(sol=model, inputs=xbc_r)
    #     g_r = -1

    #     res_u_l = ubc_l - g_l
    #     res_u_r = ubc_r - g_r

    #     res_bc = {}
    #     res_bc.update({"bc_left": res_u_l})
    #     res_bc.update({"bc_right": res_u_r})

    #     return res_bc
    
    # robin_bc
    def bc(self, sol):
        xbc = torch.ones((1, 1)).to(self.device)
        xbc_l, xbc_r = (
            torch.ones_like(xbc) * self.xmin,
            torch.ones_like(xbc) * self.xmax,
        )
        xbc_l.requires_grad, xbc_r.requires_grad = True, True

        model = sol
        # Left
        ubc_l = self.construct_sol(sol=model, inputs=xbc_l)
        du_dxl = torch.autograd.grad(
            outputs=ubc_l,
            inputs=xbc_l,
            grad_outputs=torch.ones(ubc_l.shape).to(self.device),
            create_graph=True,
        )[0]
        g_l = torch.pi * torch.sin(torch.pi * xbc_l) + self.alpha*torch.cos(torch.pi * xbc_l)

        # Right
        ubc_r = self.construct_sol(sol=model, inputs=xbc_r)
        du_dxr = torch.autograd.grad(
            outputs=ubc_r,
            inputs=xbc_r,
            grad_outputs=torch.ones(ubc_r.shape).to(self.device),
            create_graph=True,
        )[0]
        g_r = -torch.pi * torch.sin(torch.pi * xbc_r) + self.alpha*torch.cos(torch.pi * xbc_r)

        res_u_l = du_dxl * (-1) + self.alpha*ubc_l - g_l
        res_u_r = du_dxr * (1) + self.alpha*ubc_r - g_r

        res_bc = {}
        res_bc.update({"bc_left": res_u_l})
        res_bc.update({"bc_right": res_u_r})

        return res_bc

    # exact solution of x
    def ex_sol(self, inputs):
        x = inputs
        sol = torch.cos(torch.pi * x)
        return sol

    # inputs shape: [None, dims = 2]
    def val(self, sol):
        model = sol
        u_pred = self.construct_sol(sol=model, inputs=self.ref_x)
        u_exact = self.ex_sol(inputs=self.ref_x)
        err = {}
        err_u = torch.max(torch.sqrt((u_pred - u_exact) ** 2))
        # err_u = torch.sqrt(
        #     torch.mean((u_pred - u_exact) ** 2) / (torch.mean(u_exact**2) + 1e-8)
        # )
        err["u"] = err_u.to("cpu").detach().numpy()
        return err
