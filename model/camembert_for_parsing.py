import torch
import torch.nn as nn

# Biaffine Layers
class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim+1, in_dim+1))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x_dep, x_head):
        bsz, seq_len, dim = x_dep.size()
        ones = x_dep.new_ones(bsz, seq_len, 1)
        x_dep = torch.cat([x_dep, ones], dim=-1)
        x_head= torch.cat([x_head, ones], dim=-1)
        out = torch.einsum("bxi,oij,byj->boxy", x_dep, self.weight, x_head)
        return out

# MLP Layers
class MLP(nn.Module):
    """
    MLP sur n_layers, chaque layer => (Linear + activation + dropout).
    """
    def __init__(self, input_dim, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        layers=[]
        dim_in = input_dim
        for i in range(n_layers):
            dim_out = hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # On va l'appliquer en "tout-venant"
        return self.net(x)

# Biaffine Parser
class BiaffineDependencyParser(nn.Module):
    def __init__(self, input_dim, arc_dim=512, rel_dim=512, n_rels=30,
                 arc_mlp_layers=2, rel_mlp_layers=2, dropout=0.2):
        super().__init__()
        self.mlp_arc_dep = MLP(input_dim, arc_dim, n_layers=arc_mlp_layers, dropout=dropout)
        self.mlp_arc_head= MLP(input_dim, arc_dim, n_layers=arc_mlp_layers, dropout=dropout)
        self.mlp_rel_dep = MLP(input_dim, rel_dim, n_layers=rel_mlp_layers, dropout=dropout)
        self.mlp_rel_head= MLP(input_dim, rel_dim, n_layers=rel_mlp_layers, dropout=dropout)

        self.biaffine_arc = Biaffine(arc_dim, 1)
        self.biaffine_rel = Biaffine(rel_dim, n_rels)

    def forward(self, enc_out):
        arc_dep = self.mlp_arc_dep(enc_out)
        arc_head= self.mlp_arc_head(enc_out)
        rel_dep = self.mlp_rel_dep(enc_out)
        rel_head= self.mlp_rel_head(enc_out)

        arc_logits = self.biaffine_arc(arc_dep, arc_head).squeeze(1)
        rel_logits = self.biaffine_rel(rel_dep, rel_head)
        return arc_logits, rel_logits

# Camembert + Biaffine
class CamembertForParsing(nn.Module):
    def __init__(self, camembert_pretrained,
                 arc_dim=512, rel_dim=512, n_rels=30,
                 arc_mlp_layers=2, rel_mlp_layers=2,
                 dropout=0.2):
        super().__init__()
        self.camembert = camembert_pretrained.camembert
        d_model = self.camembert.config.hidden_size
        self.parser = BiaffineDependencyParser(
            input_dim=d_model,
            arc_dim=arc_dim,
            rel_dim=rel_dim,
            n_rels=n_rels,
            arc_mlp_layers=arc_mlp_layers,
            rel_mlp_layers=rel_mlp_layers,
            dropout=dropout
        )

    def forward(self, input_ids, attention_mask=None, heads=None, rels=None):
        enc_out = self.camembert(input_ids, attention_mask=attention_mask)
        arc_logits, rel_logits = self.parser(enc_out)
        loss=None
        if heads is not None and rels is not None:
            loss = self._compute_loss(arc_logits, rel_logits, heads, rels, attention_mask)
        return arc_logits, rel_logits, loss

    def _compute_loss(self, arc_logits, rel_logits, heads, rels, attention_mask):
        """
        heads[k] = index du token parent de k
        rels[k] = label relation
        attention_mask => 0 pour pad, 1 pour token
        On veut la cross entropy sur arcs => p(head=k)
           et sur rel => p(rel= gold)
        """
        bsz, seq_len, _ = arc_logits.size()
        # On va rassembler en 2D
        arc_logits_2d = arc_logits.reshape(bsz*seq_len, seq_len)
        gold_heads = heads.view(-1)

        # on veut ignorer le padding => mask
        # On ignore le token [0]
        # En gros, on ignore tout token qui a attention_mask=0
        active = attention_mask.view(-1).bool()
        # On filtre
        arc_logits_2d = arc_logits_2d[active]
        gold_heads    = gold_heads[active]

        # CrossEntropy sur arcs
        arc_loss_fct = nn.CrossEntropyLoss()
        arc_loss = arc_loss_fct(arc_logits_2d, gold_heads)

        # on index par heads
        # On fait un gather manuellement:
        batch_idx = torch.arange(bsz).unsqueeze(-1).expand(bsz, seq_len).to(rel_logits.device)
        tok_idx   = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).to(rel_logits.device)
        rel_score_for_gold_arc = rel_logits.permute(0,2,3,1)
        heads_ = heads.clone()
        heads_[heads_<0]=0
        rel_score_for_gold_arc = rel_score_for_gold_arc[batch_idx, tok_idx, heads_, :]

        rel_score_2d = rel_score_for_gold_arc.view(bsz*seq_len, -1)
        gold_rels = rels.view(-1)
        rel_score_2d = rel_score_2d[active]
        gold_rels = gold_rels[active]

        rel_loss_fct = nn.CrossEntropyLoss()
        rel_loss = rel_loss_fct(rel_score_2d, gold_rels)

        total_loss = arc_loss + rel_loss
        return total_loss
