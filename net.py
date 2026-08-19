from layer import *


class HorizonGRUDecoder(nn.Module):
    """
    Latent recurrent multi-horizon decoder (HGRU-CTX+HE, no teacher forcing).

    d0 = 0,  dh = GRUCell([zi; e_h], d_{h-1})
    Each d_h is mapped to F features by a shared linear head.
    """

    def __init__(self, in_dim: int, hidden: int, n_features: int, n_horizon: int,
                 horizon_emb_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        if horizon_emb_dim <= 0:
            raise ValueError("horizon_emb_dim must be > 0")
        self.n_features = n_features
        self.n_horizon = n_horizon
        self.hidden = hidden
        self.ctx_proj = nn.Linear(in_dim, hidden)
        self.horizon_emb = nn.Embedding(n_horizon, horizon_emb_dim)
        self.cell = nn.GRUCell(hidden + horizon_emb_dim, hidden)
        self.out = nn.Linear(hidden, n_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        # z: (B, C, N, 1)  →  (B, H*F, N, 1)
        B, C, N, _ = z.shape
        zi = self.ctx_proj(z.squeeze(-1).permute(0, 2, 1).reshape(B * N, C))
        h = z.new_zeros(B * N, self.hidden)
        steps = []
        for t in range(self.n_horizon):
            eh = self.horizon_emb.weight[t].expand(B * N, -1)
            h = self.cell(torch.cat([zi, eh], dim=-1), h)
            steps.append(self.out(self.drop(h)))
        y = torch.stack(steps, dim=1).view(B, N, self.n_horizon * self.n_features)
        return y.permute(0, 2, 1).unsqueeze(-1)


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, horizon_decoder="gru_ctx_hemb", n_horizon=0, n_features=0, gru_hidden=0, horizon_emb_dim=8, feature_graph=False, feat_gcn_depth=0):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.horizon_decoder = horizon_decoder
        self.feature_graph = feature_graph
        self.in_dim = in_dim
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        # Factorized feature-graph stem: lift -> DualMP(A_I, A_F) -> collapse F
        n_feat = n_features if n_features > 0 else in_dim
        self.n_feat_nodes = n_feat
        if feature_graph:
            if n_feat <= 0:
                raise ValueError("feature_graph=True requires n_features or in_dim > 0")
            fdep = feat_gcn_depth if feat_gcn_depth > 0 else gcn_depth
            self.gc_feat = graph_constructor(
                n_feat, n_feat, node_dim, device, alpha=tanhalpha, static_feat=None,
            )
            self.feat_idx = torch.arange(n_feat).to(device)
            # input (B,F,N,L) -> (B,1,N,F,L) -> (B,c,N,F,L)
            self.feat_lift = nn.Conv3d(1, residual_channels, kernel_size=1, bias=True)
            self.feat_dual = FactorizedDualMix(
                residual_channels, residual_channels, fdep, dropout, propalpha,
            )
            self.feat_collapse = nn.Linear(n_feat, 1, bias=True)
        else:
            self.gc_feat = None
            self.feat_lift = None
            self.feat_dual = None
            self.feat_collapse = None
            self.feat_idx = None

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        if horizon_decoder != "gru_ctx_hemb":
            raise ValueError(
                f"horizon_decoder={horizon_decoder!r} is not supported; "
                "use gru_ctx_hemb"
            )
        if n_horizon <= 0 or n_features <= 0:
            raise ValueError("HGRU-CTX+HE requires n_horizon and n_features > 0")
        gh = gru_hidden if gru_hidden > 0 else end_channels
        self.horizon_gru = HorizonGRUDecoder(
            in_dim=end_channels, hidden=gh,
            n_features=n_features, n_horizon=n_horizon,
            horizon_emb_dim=horizon_emb_dim, dropout=dropout,
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)

    def _feature_stem(self, input, adp):
        """
        input: (B, F, N, L) -> (B, C, N, L) via lift + FactorizedDualMix + collapse.
        Shares industry adjacency `adp` with the main mixprop stack; A_F is learned.
        """
        B, Fdim, N, L = input.shape
        assert Fdim == self.n_feat_nodes, (
            f"feature_graph expects in_dim/F={self.n_feat_nodes}, got {Fdim}"
        )
        # (B, F, N, L) -> (B, 1, N, F, L)
        x = input.permute(0, 2, 1, 3).unsqueeze(1)
        x = self.feat_lift(x)  # (B, C, N, F, L)
        a_f = self.gc_feat(self.feat_idx)
        if adp is None:
            adp = torch.eye(N, device=input.device, dtype=input.dtype)
        x = self.feat_dual(x, adp, a_f)
        # collapse F: (B,C,N,F,L) -> (B,C,N,L)
        x = self.feat_collapse(x.permute(0, 1, 2, 4, 3)).squeeze(-1)
        return x

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        adp = None
        if self.gcn_true or self.feature_graph:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        if self.feature_graph:
            x = self._feature_stem(input, adp)
        else:
            x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.horizon_gru(x)
