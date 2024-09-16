import torch
from feature_extract.configs.custom_config import load_config
from slowfast.utils.parser import parse_args
from slowfast.models.head_helper import ResNetBasicHead


class ResNetBasicHead(ResNetBasicHead):
    # Overwrite function to return features
    #原本的forward output为class长度，现在改为C长度，C为channel
    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        # save features
        feat = x.clone().detach()
        # flatten the features tensor
        feat = feat.mean(3).mean(2).reshape(feat.shape[0], -1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x, feat



if __name__ == "__main__":
    print("debug this file ...")


    args = parse_args()
    cfg = load_config(args)
    slow = torch.randn((12,3,32,256,256))
    fast = torch.randn((12,3,32,256,256))
    inputs  = [slow,fast]
    resnethead = ResNetBasicHead(dim_in=3,num_classes=1000,pool_size= None,cfg=cfg)
    x = resnethead.forward(inputs)

    print("debug finish ...")




