import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)) #output: [sin(2^freq *x), cos(2^freq *x)]
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder = Embedder(**embed_kwargs)
    return embedder, embedder.out_dim

class NeRF(nn.Module):
    """
        NeRF Network.

        Input:
            embedded 3D location and 2D view direction
        Output:
            [rgb,density]

            rgb: the predicted rgb value of a point from a given view

            density: volume density, can be interpreted as the differential probability of a
                     ray terminating at an infinitesimal particle at location x.
    """
    def __init__(self, D=8, W=256, input_ch_pts=3, input_ch_views=3, skips=[4]):
        super(NeRF, self).__init__()
        self.skips = skips
        self.input_ch_pts = input_ch_pts
        self.input_ch_views = input_ch_views

        self.mlps = nn.ModuleList(
            [nn.Linear(input_ch_pts, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch_pts, W) for i
                                            in
                                            range(D - 1)]) #like resnet to fuse deep/shallow info
        
        self.rgb_feature_linear = nn.Linear(W, W)
        self.rgb_feature_linears = nn.ModuleList([nn.Linear(W + input_ch_views, W // 2)])

        self.density_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        input_pts, input_dirs = torch.split(x, [self.input_ch_pts, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.mlps):
            h = self.mlps[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)
        
        rgb_feature = self.rgb_feature_linear(h)
        rgb_feature = torch.cat([rgb_feature, input_dirs], -1) # add 2D viewdir_info
        for i, l in enumerate(self.rgb_feature_linears):
            rgb_feature = self.rgb_feature_linears[i](rgb_feature)
            rgb_feature = F.relu(rgb_feature)


        density = self.density_linear(h) #output of the middle layer, irrelated with the direction information
        rgb = self.rgb_linear(rgb_feature)

        # check here, TODO
        #rgb = torch.sigmoid(rgb)
        #density = torch.sigmoid(density)

        outputs = torch.cat([rgb, density], -1)
        return outputs