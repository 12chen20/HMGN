import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from mgkap import MGKAP



class Sparsemax(nn.Module):

    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        input = input - input.max(dim=self.dim, keepdim=True)[0]
        zs = torch.sort(input, descending=True, dim=self.dim)[0]
        range_ = torch.arange(1, zs.size(self.dim) + 1, device=input.device, dtype=input.dtype)


        if input.dim() == 2:
            range_ = range_.view(1, -1)
        else:
            shape = [1 if i != self.dim else -1 for i in range(input.dim())]
            range_ = range_.view(shape)

        bound = 1 + range_ * zs
        cumsum_zs = zs.cumsum(dim=self.dim)
        is_gt = (bound > cumsum_zs).type(input.dtype)
        k = (is_gt * range_).max(dim=self.dim, keepdim=True)[0]
        tau = (cumsum_zs.gather(self.dim, k.long() - 1) - 1) / k
        output = torch.clamp(input - tau, min=0)
        return output

# =========================
#  HAMS
# =========================
class HAMSConv(nn.Module):
    """
    Heterogeneity-Aware Multi-Scale Convolutional Module (HAMS)

    """
    def __init__(self, dim, n_neighbor, kernel_sizes=[1, 3, 5, 7]):
        super(HAMSConv, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.n_neighbor = n_neighbor
        self.dim = dim


        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=n_neighbor, out_channels=1,
                kernel_size=(k, 3), padding=(k // 2, 1)
            ) for k in kernel_sizes
        ])


        self.fc_alpha = nn.Linear(dim, len(kernel_sizes))
        self.fc_tau = nn.Linear(dim, 1)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sparsemax = Sparsemax(dim=-1)

        self.lambda_res = nn.Parameter(torch.tensor(0.2))

    def forward(self, P_u, u_vec):
        """
        P_u: [batch, n_neighbor, dim, 1]
        u_vec: [batch, dim]
        """
        B = u_vec.size(0)


        h = self.activation(u_vec)
        tau = torch.sigmoid(self.fc_tau(h)) + 1e-5
        alpha_logits = self.fc_alpha(h) / tau

        alpha = self.sparsemax(alpha_logits)




        conv_outputs = []
        for i, conv in enumerate(self.convs):
            conv_out = conv(P_u).squeeze()
            conv_outputs.append(conv_out * alpha[:, i].unsqueeze(1))

        multi_scale_out = torch.stack(conv_outputs, dim=0).sum(dim=0)


        residual = P_u.mean(dim=1).squeeze()                   # [B, dim]
        output = multi_scale_out + self.lambda_res * residual

        return output  # [B, dim]


# =========================
#  PIFSA-GNN
# =========================
class HMGN_GNN(nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device, UAndI):
        super(HMGN_GNN, self).__init__()


        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.device = device
        self.kg = kg


       
        self.aggregator = MGKAP(self.batch_size, self.dim, args.aggregator,
                        num_heads=4, num_layers=2, dropout=0.1)



        self.usr = nn.Embedding(num_user, self.dim)
        self.ent = nn.Embedding(num_ent, self.dim)
        self.rel = nn.Embedding(num_rel, self.dim)


        self.hams_conv = HAMSConv(dim=self.dim, n_neighbor=self.n_neighbor)
        self.baseline_conv = nn.Conv2d(in_channels=self.n_neighbor, out_channels=1,
                              kernel_size=(3, 3), padding=(1, 1))


        self.UAndI_tensor = torch.zeros((self.num_user, self.n_neighbor), dtype=torch.long)
        for user_id, item_ids in UAndI.items():
            items = item_ids[:self.n_neighbor]
            if len(items) < self.n_neighbor:
                items += [0] * (self.n_neighbor - len(items))
            self.UAndI_tensor[user_id] = torch.LongTensor(items)
        self.UAndI_tensor = self.UAndI_tensor.to(device)


        self._gen_adj()
        self.adj_ent = self.adj_ent.to(self.device)
        self.adj_rel = self.adj_rel.to(self.device)


    def _gen_adj(self):
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in tqdm(self.kg, total=len(self.kg), desc='Generate KG adjacency matrix:'):
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])


    def forward(self, u, v):
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size


        user_and_item_adj = self.UAndI_tensor[u]                   # [batch, n_neighbor]
        UAndIEmmbbing = self.ent(user_and_item_adj)                # [batch, n_neighbor, dim]
        u = u.view((-1, 1))
        v = v.view((-1, 1))


        user_base = self.usr(u).squeeze(1)                         # [batch, dim]
        user_embeddings = user_base.unsqueeze(1) * UAndIEmmbbing   # [batch, n_neighbor, dim]
        user_embeddings = user_embeddings.unsqueeze(dim=3)         # [batch, n_neighbor, dim, 1]


        user_embeddings = self.hams_conv(user_embeddings, user_base)  # [batch, dim]




        entities, relations = self._get_neighbors(v)
        item_embeddings = self._aggregate(user_embeddings, entities, relations)


        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return torch.sigmoid(scores)


    def _get_neighbors(self, v):
        v = v.view(-1)
        neighbor_entities = self.adj_ent[v]
        neighbor_relations = self.adj_rel[v]

        entities = [v, neighbor_entities]
        relations = [neighbor_relations]
        return entities, relations


    def _aggregate(self, user_embeddings, entities, relations):
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        vector = self.aggregator(
            self_vectors=entity_vectors[0],
            neighbor_vectors=entity_vectors[1],
            neighbor_relations=relation_vectors[0],
            user_embeddings=user_embeddings
        )
        return vector

