import torch
from algos.base_algo import Base_Algo

class Unconditional(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None):
        super().__init__(model, H_funcs, sigma_0, cls_fn)

    @ torch.enable_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, noise='ddpm', classes=None):
        #xt.requires_grad_(True)
        if self.cls_fn == None:
            et = self.model(xt, t)
        else:
            et = self.model(xt, t, self.classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(xt,t,self.classes)
        if et.size(1) == 6:
            et = et[:, :3]
        # et = et.clip(-1, 1)
        self.et = et
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_t = x0_t.clip(-1, 1)
        if noise == 'ddpm':
            c1 = ((1-at[0,0,0,0]/at_next[0,0,0,0]) * (1-at_next[0,0,0,0])/(1-at[0,0,0,0])).sqrt()
        elif noise == 'ddim':
            c1 = 0
        else:
            raise ValueError("Unsupported noise type: {}".format(noise))
        c2 = (1-at_next[0,0,0,0] - c1**2).sqrt()
        #print(t, self.noise_seq[t.item()][0,0,0,0])
        #add_up = c1 * self.noise_seq[t.item()] + c2 * et
        add_up = (1 - at_next).sqrt() * et
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next