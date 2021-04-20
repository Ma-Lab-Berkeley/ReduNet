import torch
import torch.nn as nn



class ReduLayer(nn.Module):
    def __init__(self):
        super(ReduLayer, self).__init__()

    def __name__(self):
    	return "ReduNet"

    def forward(self, Z):
        raise NotImplementedError

    def zero(self):
        state_dict = self.state_dict()
        state_dict['E.weight'] = torch.zeros_like(self.E.weight)
        for j in range(self.num_classes):
            state_dict[f'Cs.{j}.weight'] = torch.zeros_like(self.Cs[j].weight)
        self.load_state_dict(state_dict)

    def init(self, X, y):
        gam = self.compute_gam(X, y)
        E = self.compute_E(X)
        Cs = self.compute_Cs(X, y)
        self.set_params(E, Cs, gam)

    def update_old(self, X, y, tau):
        E = self.compute_E(X).to(X.device)
        Cs = self.compute_Cs(X, y).to(X.device)
        state_dict = self.state_dict()
        ref_E = self.E.weight
        ref_Cs = [self.Cs[j].weight for j in range(self.num_classes)]
        new_E = ref_E + tau * (E - ref_E)
        new_Cs = [ref_Cs[j] + tau * (Cs[j] - ref_Cs[j]) for j in range(self.num_classes)]
        state_dict['E.weight'] = new_E
        for j in range(self.num_classes):
            state_dict[f'Cs.{j}.weight'] = new_Cs[j]
        self.load_state_dict(state_dict)

    def update(self, X, y, tau):
        E_ref, Cs_ref = self.get_params()
        # gam = self.init_gam(X, y)
        E_new = self.compute_E(X).to(X.device)
        Cs_new = self.compute_Cs(X, y).to(X.device)
        E_update = E_ref + tau * (E_new - E_ref)
        Cs_update = [Cs_ref[j] + tau * (Cs_new[j] - Cs_ref[j]) for j in range(self.num_classes)]
        self.set_params(E_update, Cs_update)

    def set_params(self, E, Cs, gam=None):
        state_dict = self.state_dict()
        assert self.E.weight.shape == E.shape, f'E shape does not match: {self.E.weight.shape} and {E.shape}'
        state_dict['E.weight'] = E
        for j in range(self.num_classes):
            assert self.Cs[j].weight.shape == Cs[j].shape, f'Cj shape does not match'
            state_dict[f'Cs.{j}.weight'] = Cs[j]
        if gam is not None:
            assert self.gam.shape == gam.shape, 'gam shape does not match'
            state_dict['gam'] = gam
        self.load_state_dict(state_dict)

    def get_params(self):
        E = self.E.weight
        Cs = [self.Cs[j].weight for j in range(self.num_classes)]
        return E, Cs
    
