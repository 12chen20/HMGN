import torch
import torch.nn as nn
import torch.nn.functional as F


class MGKAP(nn.Module):


    def __init__(self, batch_size, dim, aggregator='sum',
                 num_heads=4, num_layers=2, dropout=0.3):
        super(MGKAP, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.aggregator = aggregator
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)


        if aggregator == 'concat':
            self.weights = nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = nn.Linear(dim, dim, bias=True)

        self.multihead_layers = nn.ModuleList([
            MultiHeadInteractionAttention(dim, num_heads)
            for _ in range(num_layers)
        ])


        self.W_prop = nn.ModuleList([
            nn.Linear(dim, dim, bias=True)
            for _ in range(num_layers)
        ])
        self.W_self = nn.ModuleList([
            nn.Linear(dim, dim, bias=True)
            for _ in range(num_layers)
        ])


        self.W_layer = nn.Linear(2 * dim, dim)
        self.w_layer = nn.Parameter(torch.randn(dim))


    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):

        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size


        neighbors_agg = self._multi_granular_propagation(
            neighbor_vectors, neighbor_relations, user_embeddings
        )  # [B, dim]


        if self.aggregator == 'sum':
            output = self_vectors + neighbors_agg
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
        else:
            output = torch.max(self_vectors, neighbors_agg)[0]  # [B, dim]

        return self.weights(output)


    def _multi_granular_propagation(self, neighbor_vectors, neighbor_relations, user_embeddings):

        B = user_embeddings.size(0)
        v_l = neighbor_vectors.mean(dim=1)
        v_layers = [v_l]

        for l in range(self.num_layers):

            m_r_u_e, m_e_u_r = self.multihead_layers[l](
                user_embeddings, neighbor_vectors, neighbor_relations
            )  # [B, n_neighbor, dim] ×2


            propagated = (m_r_u_e + m_e_u_r).sum(dim=1)  # [B, dim]
            v_next = torch.tanh(
                self.W_prop[l](propagated) + self.W_self[l](v_layers[-1])
            )
            v_next = self.dropout(v_next)
            v_layers.append(v_next)


        return self._hierarchical_fusion(user_embeddings, v_layers)

    def _hierarchical_fusion(self, user_embeddings, v_layers):

        B = user_embeddings.size(0)
        L = len(v_layers)

        # [B, L, dim]
        v_stack = torch.stack(v_layers, dim=1)
        user_expand = user_embeddings.unsqueeze(1).expand(B, L, self.dim)

        fusion_input = torch.cat([user_expand, v_stack], dim=-1)
        fusion_hidden = torch.tanh(self.W_layer(fusion_input))
        fusion_score = torch.matmul(fusion_hidden, self.w_layer)
        gamma = F.softmax(fusion_score, dim=1).unsqueeze(-1)

        v_fused = torch.sum(gamma * v_stack, dim=1)
        return v_fused



class MultiHeadInteractionAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(MultiHeadInteractionAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.scale = self.d_k ** -0.5

        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(2 * dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.ones(num_heads))
        self.beta = nn.Parameter(torch.ones(num_heads))

    def forward(self, user_embeddings, entity_vectors, relation_vectors):

        B, N, D = entity_vectors.size()
        H, dk = self.num_heads, self.d_k


        Q = self.W_Q(user_embeddings).view(B, H, dk)
        K_input = torch.cat([relation_vectors, entity_vectors], dim=-1)
        K = self.W_K(K_input).view(B, N, H, dk)
        V_e = self.W_V(entity_vectors).view(B, N, H, dk)
        V_r = self.W_V(relation_vectors).view(B, N, H, dk)


        att_scores = torch.einsum('bhd,bnhd->bnh', Q, K) * self.scale
        att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)
        att_weights = self.dropout(att_weights)


        m_r_u_e = (att_weights * V_e).permute(0, 2, 1, 3).reshape(B, N, D)
        m_e_u_r = (att_weights * V_r).permute(0, 2, 1, 3).reshape(B, N, D)


        m_r_u_e = m_r_u_e * (self.alpha.sum() / H)
        m_e_u_r = m_e_u_r * (self.beta.sum() / H)
        return m_r_u_e, m_e_u_r
