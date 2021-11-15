import torch
from torch import nn as nn

from utils.util import opt_get
from models.modules import ActNorms, Permutations, AffineCouplings

import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation

class FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels=None, flow_permutation='invconv', flow_coupling='Affine', LRvsothers=True,
                 actnorm_scale=1.0, LU_decomposed=False, opt=None):
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling

        # 1. actnorm
        self.actnorm = ActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute # todo: maybe hurtful for downsampling; presever the structure of downsampling
        if self.flow_permutation == "invconv":
            self.permute = Permutations.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        elif self.flow_permutation == "none":
            self.permute = None

        # 3. coupling
        if self.flow_coupling == "AffineInjector":
            self.affine = AffineCouplings.AffineCouplingInjector(in_channels=in_channels, cond_channels=cond_channels, opt=opt)
        elif self.flow_coupling == "noCoupling":
            pass
        elif self.flow_coupling == "Affine":
            self.affine = AffineCouplings.AffineCoupling(in_channels=in_channels, cond_channels=cond_channels, opt=opt)
        elif self.flow_coupling == "Affine3shift":
            self.affine = AffineCouplings.AffineCoupling3shift(in_channels=in_channels, cond_channels=cond_channels, LRvsothers=LRvsothers, opt=opt)

    def forward(self, z, u=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, logdet)
        else:
            return self.reverse_flow(z, u)

    def normal_flow(self, z, u=None, logdet=None):
        # 1. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        if self.permute is not None:
            z, logdet = self.permute( z, logdet=logdet, reverse=False)

        # 3. coupling
        z, logdet = self.affine(z, u=u, logdet=logdet, reverse=False)

        return z, logdet

    def reverse_flow(self, z, u=None, logdet=None):
        # 1.coupling
        z, _ = self.affine(z, u=u, reverse=True)

        # 2. permute
        if self.permute is not None:
            z, _ = self.permute(z, reverse=True)

        # 3. actnorm
        z, _ = self.actnorm(z, reverse=True)

        return z, logdet




def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class SRFLOWFlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in SRFLOWFlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                SRFLOWFlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        if self.flow_coupling == "bentIdentityPreAct":
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)

        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        need_features = self.affine_need_features()

        # 3. coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features()

        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
