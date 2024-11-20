from typing import Optional,Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

import midi.utils as utils
from midi.diffusion import diffusion_utils
from midi.models.layers import Xtoy, Etoy, SE3Norm, PositionsMLP, masked_softmax, EtoX, SetNorm, GraphNorm, PositionsAttentionMLP

class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int,df: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dim_fff: int = 256, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, last_layer=False) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, df, n_head, last_layer=last_layer)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        # self.normX1 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        # self.normX2 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps, **kw)
        

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        # self.normE1 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        # self.normE2 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)
            
            self.lin_f1 = Linear(df, dim_fff, **kw)
            self.lin_f2 = Linear(dim_fff, df, **kw)
            self.norm_f1 = LayerNorm(df, eps=layer_norm_eps, **kw)
            self.norm_f2 = LayerNorm(df, eps=layer_norm_eps, **kw)
            self.dropout_f1 = Dropout(dropout)
            self.dropout_f2 = Dropout(dropout)
            self.dropout_f3 = Dropout(dropout)
            
            self.norm_p = SE3Norm(eps=layer_norm_eps, **kw)
            

        self.activation = F.relu
    def forward(self, features: utils.PlaceHolder):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        X = features.X
        E = features.E
        y = features.y
        pos = features.pos
        node_mask = features.node_mask
        p = features.pharma_coord
        f = features.pharma_feat
        pharma_mask = features.pharma_mask        
        
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        p_mask = pharma_mask.unsqueeze(-1)
        # e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        # e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        newX, newE, new_y, new_f, vel, vel_p = self.self_attn(X, E, y, pos,f, p, node_mask=node_mask, pharma_mask=pharma_mask)

        newX_d = self.dropoutX1(newX)
        # X = self.normX1(X + newX_d, x_mask)
        X = self.normX1(X + newX_d)
        
        new_pos = self.norm_pos1(vel, x_mask) + pos 
        
        if torch.isnan(new_pos).any():
            raise ValueError("NaN in new_pos")

        newE_d = self.dropoutE1(newE)
        # E = self.normE1(E + newE_d, e_mask1, e_mask2)
        E = self.normE1(E + newE_d)

        if not self.last_layer:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)
            
            #new_f_d = self.dropout_f1(new_f)
            #f = self.norm_f1(f + new_f_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        # X = self.normX2(X + ff_outputX, x_mask)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        # E = self.normE2(E + ff_outputE, e_mask1, e_mask2)
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)

            
            #ff_output_f = self.lin_f2(self.dropout_f2(self.activation(self.lin_f1(new_f))))
            #ff_output_f = self.dropout_f3(ff_output_f)
            #f = self.norm_f2(f + ff_output_f)
            
            #new_p = self.norm_p(vel_p, p_mask) + p
            
            
            #new_p = new_p * p_mask
            #f = f * p_mask
            

        out = utils.PlaceHolder(X=X, E=E, y=y, pos=new_pos, charges=None, node_mask=node_mask, 
                                pharma_coord=p, pharma_feat=f, pharma_mask=pharma_mask).mask()

        return out


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, df, n_head, last_layer=False):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        #self.df = int(dx / n_head)
        self.df = df
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        # self.x_e_add = Linear(dx, de)
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)
        
        
        # # pharma_feat to E
        self.f_e_mul1 = Linear(df, de)
        self.f_e_mul2 = Linear(df, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_norm_pos1 = Linear(1, de)
        self.lin_norm_pos2 = Linear(1, de)
        

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)
        # self.lin_dist2 = Linear(dx, dx)
        
        self.dist_add_e_p = Linear(de, de)
        self.dist_mul_e_p = Linear(de, de)
        
        
                
        # #pharma_coord encoding
        self.lin_dist1_p = Linear(2, de)
        self.lin_norm_p = Linear(1, de)
        self.lin_norm_p2 = Linear(1, de)
        
        self.dist_add_e_p = Linear(de, de)
        self.dist_mul_e_p = Linear(de, de)
        

        # Attention
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)
        
        # self.k_f = Linear(df, df)
        self.q_f = Linear(df, df)
        # self.v_f = Linear(df, df)
        self.a_f = Linear(df, n_head, bias=False)
        # self.out_f = Linear(dx * n_head, dx)
        
        self.k_pos = Linear(3, de)
        self.q_pos = Linear(3, de)
        self.v_pos = Linear(3, de)
        self.a_pos = Linear(de, n_head, bias=False)
        self.d_pos = Linear(1, n_head)
        self.out_pos = Linear(de * n_head, 3)
        

        # Incorporate e to x
        # self.e_att_add = Linear(de, n_head)
        self.e_att_mul = Linear(de, n_head)
        
        self.f_att_mul = Linear(df, n_head)

        self.pos_att_mul = Linear(de, n_head)
        
        self.p_att_mul = Linear(de, n_head)

        self.e_x_mul = EtoX(de, dx)
        
        #self.f_x_mul = EtoX(df, dx)

        self.pos_x_mul = EtoX(de, dx)
        
        self.p_x_mul = EtoX(de, dx)


        # FiLM y to E
        self.y_e_mul = Linear(dy, de)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)       # Unused, but needed to load old checkpoints

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)
        
        # # FiLM f to X
        self.f_x_mul = Linear(df, dx)
        self.f_x_add = Linear(df, dx)
        
        
        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)
            self.distp_y = Etoy(de, dy)
            self.f_y = Xtoy(df, dy)
            
            self.f_f = Linear(df, df)
            # self.x_f = Xtoy(dx, df)
            # self.e_f = Etoy(de, df)
            # self.dist_f = Etoy(de, df)
            # self.dist_fp = Etoy(de, df)
            # self.y_f = Linear(dy, df)
        
                        

        # Process_pos
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)          # For EGNN v3: map to pi, pj
        
        self.e_p1 = Linear(de, de, bias=False)
        self.e_p2 = Linear(de, 1, bias=False)          # For EGNN v3: map to pi, pj 

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        
        self.pos_att = PositionsAttentionMLP(de)
        
        if not last_layer:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))
            self.f_out = nn.Sequential(nn.Linear(df, df), nn.ReLU(), nn.Linear(df, df))

    def forward(self, X, E, y, pos, f, p, node_mask, pharma_mask):
        """ :param X: bs, n, d        node features
            :param E: bs, n, n, d     edge features
            :param y: bs, dz           global features
            :param pos: bs, n, 3
            :param node_mask: bs, n
            :return: newX, newE, new_y with the same shape. """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        p_mask = pharma_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        pe_mask1 = p_mask.unsqueeze(2)
        pe_mask2 = p_mask.unsqueeze(1)

        # 0. Create a distance matrix that can be used later
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)         # bs, n, 1
        normalized_pos = pos / (norm_pos + 1e-7)                 # bs, n, 3

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1).float()
        cosines = torch.sum(normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2), dim=-1, keepdim=True)
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)

        norm1 = self.lin_norm_pos1(norm_pos)             # bs, n, de
        norm2 = self.lin_norm_pos2(norm_pos)             # bs, n, de
        dist1 = F.relu(self.lin_dist1(pos_info) + norm1.unsqueeze(2) + norm2.unsqueeze(1)) * e_mask1 * e_mask2
        
        
        p = p * p_mask
        norm_p = torch.norm(p, dim=-1, keepdim=True)         # bs, n, 1
        normalized_p = p / (norm_p + 1e-7)                 # bs, n, 3
        
        pairwise_dist_p = torch.cdist(p, p).unsqueeze(-1).float()
        cosines_p = torch.sum(normalized_p.unsqueeze(1) * normalized_p.unsqueeze(2), dim=-1, keepdim=True)
        pos_info_p = torch.cat((pairwise_dist_p, cosines_p), dim=-1)
        
        norm1_p = self.lin_norm_p(norm_p)             # bs, n, de
        norm2_p = self.lin_norm_p2(norm_p)             # bs, n, de
        dist1_p = F.relu(self.lin_dist1_p(pos_info_p) + norm1_p.unsqueeze(2) + norm2_p.unsqueeze(1)) * pe_mask1 * pe_mask2

        # 1. Process E
        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y_x = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2
        
        
        
        # 1.1.2 Incorporate f
        f_e_mul1 = self.f_e_mul1(f) * p_mask
        f_e_mul2 = self.f_e_mul2(f) * p_mask
        Y_f = Y * f_e_mul1.unsqueeze(1) * f_e_mul2.unsqueeze(2) * pe_mask1 * pe_mask2
        
        Y = Y_x + Y_f

        # 1.2. Incorporate distances
        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y_pos = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2   # bs, n, n, dx
        
        
        # 1.2.2 Incorporate distances
        dist_add_p = self.dist_add_e_p(dist1_p)
        dist_mul_p = self.dist_mul_e_p(dist1_p)
        Y_p = (Y + dist_add_p + Y * dist_mul_p) * pe_mask1 * pe_mask2   # bs, n, n, dx
        
        Y = Y_pos + Y_p

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(E) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(Eout, e_mask1 * e_mask2)

        # 2. Process the node features
        Q = (self.q(X) * x_mask).unsqueeze(2)          # bs, 1, n, dx
        K = (self.k(X) * x_mask).unsqueeze(1)          # bs, n, 1, dx
        prod = Q * K / math.sqrt(Y.size(-1))   # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2   # bs, n, n, n_head

        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a
        
        # incorporate pharma features
        #f_x_mul = self.f_att_mul(f).unsqueeze(2).expand(-1, -1, n, -1)
        #a = a + f_x_mul * a

        # 2.2 Incorporate position features
        pos_x_mul = self.pos_att_mul(dist1)
        a = a + pos_x_mul * a
        a_pos = a * e_mask1 * e_mask2
        
        # 2.2.2 Incorporate pharma position features
        p_x_mul= self.p_att_mul(dist1_p)
        a = a + p_x_mul * a
        a_p = a * pe_mask1 * pe_mask2
        
        a = a_pos + a_p


        ## fetas gate
        Q_f = (self.q_f(f) * p_mask).unsqueeze(2)          # bs, 1, n, dx
        prod_f = Q_f * K / math.sqrt(Y.size(-1))   # bs, n, n, dx
        a = a + (self.a_f(prod_f) * pe_mask1 * pe_mask2)   # bs, n, n, n_head
        
        
        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        weighted_V = alpha * V                                  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)                      # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, n_head x dx
        weighted_V = self.out(weighted_V) * x_mask              # bs, n, dx
        
        #Feats attention
        # 
        # K_f = (self.k_f(f) * p_mask).unsqueeze(1)          # bs, n, 1, dx
        # 
        # a_f = self.a_f(prod_f) * pe_mask1 * pe_mask2   # bs, n, n, n_head
        
        # softmax_mask_f = pe_mask2.expand(-1, n, -1, self.n_head)
        # alpha = masked_softmax(a_f, softmax_mask_f, dim=2).unsqueeze(-1)  # bs, n, n, n_head
        # V_f = (self.v_f(X) * x_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        # weighted_V_f = alpha * V_f                                  # bs, n, n, n_heads, dx
        # weighted_V_f = weighted_V_f.sum(dim=2)                      # bs, n, n_head, dx
        # weighted_V_f = weighted_V_f.flatten(start_dim=2)            # bs, n, n_head x dx
        # weighted_V_f = self.out_f(weighted_V_f) * p_mask              # bs, n, dx
        
        # weighted_V = weighted_V + weighted_V_f

        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V

        pos_x_mul = self.pos_x_mul(dist1, e_mask2)
        weighted_V = weighted_V + pos_x_mul * weighted_V
        
        # Incorporate f to X
        fx1 = self.f_x_add(f)
        fx2 = self.f_x_mul(f)
        weighted_V = weighted_V * (fx2 + 1) + fx1
        
        p_x_mul = self.p_x_mul(dist1_p, pe_mask2)
        weighted_V = weighted_V + p_x_mul * weighted_V
        

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)                     # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(Xout, x_mask)

        # Process y based on X and E
        if self.last_layer:
            y_out = None
            f_out = None
            p_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            new_y = y + x_y + e_y + dist_y
            y_out = self.y_out(new_y)               # bs, dy
            
            
            #f = self.f_f(f)
            # e_f = self.e_f(Y, pe_mask1, pe_mask2).unsqueeze(1)
            # x_f = self.x_f(newX, p_mask).unsqueeze(1)
            # y_f = self.y_f(y).unsqueeze(1)
            # dist_fp = self.dist_fp(dist1_p, pe_mask1, pe_mask2).unsqueeze(1)
            # dist_fpos = self.dist_f(dist1, e_mask1, e_mask2).unsqueeze(1)
            # new_f = f + x_f + e_f + dist_fp + dist_fpos + y_f
            #f_out = self.f_out(f)               # bs, dx
            
            p_1 = p.unsqueeze(1).expand(-1, n, -1, -1) 
            p_2 = p.unsqueeze(2).expand(-1, -1, n, -1)
            delta_p = p_2 - p_1
            messages_p = self.e_p2(F.relu(self.e_p1(Y)))       # bs, n, n, 1, 2
            vel_p = (messages_p * delta_p).sum(dim=2) * p_mask
            vel_p, _ = utils.remove_mean_with_mask(vel_p, pharma_mask)

        # Update the positions
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)              # bs, 1, n, 3
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)              # bs, n, 1, 3
        delta_pos = pos2 - pos1                                    # bs, n, n, 3
                 

        messages = self.e_pos2(F.relu(self.e_pos1(Y)))       # bs, n, n, 1, 2
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        
        
        #Feats attention
        Q_pos = (self.q_pos(vel) * x_mask).unsqueeze(2)       
        K_pos = (self.k_pos(vel) * x_mask).unsqueeze(1)        
        pred_pos = Q_pos * K_pos / math.sqrt(Y.size(-1))
        a_pos = self.a_pos(pred_pos) * e_mask1 * e_mask2    #bs, n, n, n_head
        delta_p_masked = torch.norm(delta_p, dim=-1, keepdim=True)  * pe_mask1 * pe_mask2
        d_pos = self.d_pos(delta_p_masked) * pe_mask1 * pe_mask2
        a_pos = a_pos - d_pos

        
        softmax_mask_pos = e_mask2.expand(-1, n, -1, self.n_head)
        alpha_pos = masked_softmax(a_pos, softmax_mask_pos, dim=2).unsqueeze(-1)  # bs, n, n, n_head
        V_pos = (self.v_pos(vel) * x_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        weighted_V_pos = alpha_pos * V_pos                                  # bs, n, n, n_heads, dx
        weighted_V_pos = weighted_V_pos.sum(dim=2)                      # bs, n, n_head, dx
        weighted_V_pos = weighted_V_pos.flatten(start_dim=2)            # bs, n, n_head x dx
        weighted_V_pos = self.out_pos(weighted_V_pos) * x_mask 
        
        #vel = vel + weighted_V_pos
        
        
        
        vel, _ = utils.remove_mean_with_mask(weighted_V_pos, node_mask)
        return Xout, Eout, y_out, f, vel, vel_p


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, input_dims: utils.PlaceHolder, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: utils.PlaceHolder):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges
        self.input_dims = input_dims
        self.outdim = output_dims

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()
        
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges,
                                                hidden_mlp_dims['X']), act_fn_in, 
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        
        self.mlp_in_feats = nn.Sequential(nn.Linear( input_dims.pharma_feat , hidden_mlp_dims['f']), act_fn_in, 
                                      nn.Linear(hidden_mlp_dims['f'], hidden_dims['df']), act_fn_in)
        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims.E, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims.y, hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims['pos'])
        
        self.mlp_in_pharma_pos = PositionsMLP(hidden_mlp_dims['pos'])

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            df =hidden_dims['df'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            dim_fff = hidden_dims['dim_ffF'],
                                                            last_layer=False)     # needed to load old checkpoints
                                                            # last_layer=(i == n_layers - 1))
                                        for i in range(n_layers)])
        
        

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims.X + output_dims.charges))
        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims.E))
        # self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
        #                                nn.Linear(hidden_mlp_dims['y'], output_dims.y))
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims['pos'])
    

    def forward(self, data: utils.PlaceHolder):

        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask

    
        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        
        
        data.pharma_feat = data.pharma_feat * data.pharma_mask.unsqueeze(-1)
        
        X = torch.cat((data.X, data.charges), dim= -1)




        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]
        E_to_out = data.E[..., :self.out_dim_E]
        y_to_out = data.y[..., :self.out_dim_y]

        #pos_masked = torch.where(data.pharma_mask.unsqueeze(-1) > 0, data.pharma_coord, data.pos)



        #new_X = torch.cat((self.mlp_in_X(X), self.mlp_in_feats(data.pharma_feat)), dim = -1)
        new_X = self.mlp_in_X(X)
        new_E = self.mlp_in_E(data.E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        
        
        f = self.mlp_in_feats(data.pharma_feat) * data.pharma_mask.unsqueeze(-1)
        p  = self.mlp_in_pharma_pos(data.pharma_coord, data.pharma_mask) * data.pharma_mask.unsqueeze(-1)
        
        features = utils.PlaceHolder(X=new_X, E=new_E, y=self.mlp_in_y(data.y), charges=None,
                                     pos= self.mlp_in_pos(data.pos, node_mask), node_mask=node_mask,
                                     pharma_coord=p, pharma_feat=f, pharma_mask=data.pharma_mask).mask()
        

        for layer in self.tf_layers:
            features = layer(features)

        X = self.mlp_out_X(features.X)
        E = self.mlp_out_E(features.E)
        # y = self.mlp_out_y(features.y)
        pos = self.mlp_out_pos(features.pos, node_mask )


        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        # y = y + y_to_out
        y = y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        final_X = X[:, :, :self.out_dim_X]
        charges = X[:, : ,self.out_dim_X:self.out_dim_X+self.out_dim_charges]
        out = utils.PlaceHolder(pos=pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask, 
                                pharma_coord=data.pharma_coord, pharma_feat=data.pharma_feat, 
                                pharma_mask=data.pharma_mask).mask()
        return out


