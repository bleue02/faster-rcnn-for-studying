from __future__ import absolute_import
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
import inspect
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30] # features: Convolution layer, list-> for slicing
    classifier = model.classifier # classifiier: Fully Connected layer

    classifier = list(classifier)
    del classifier[6] # linear(in_feature:4096, out_feature(numclass): 1000) <-- 을 제외한 0~5 nn.Sequential 내부 layers가져온다.
    if not opt.use_drop: # # del drop out layers
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier) # 나머지 nn.Sequential의 layers를 unpacking 후--> classifier에 pass

    # freeze top4 conv
    for layer in features[:10]: # [:30] -> 0~29개의 layers중 0~9개의 layers만 frezz고정하고 나머지 layers의 한 해서는 backpropagation 허용(pytorch=True)
        for p in layer.parameters():
            p.requires_grad = False # requires_grad: do backpropagation(do not update parameter)

    return nn.Sequential(*features), classifier
    # *features는 리스트를 언패킹하여 각 모듈을 Sequential 생성자에 전달하기 위해 사용됨
    # classifier는 이미 Sequential 객체이므로 별도의 언패킹(*)가 필요하지 않음
    # 즉 features는 0~29개의 layers를 반환되지만 0~9개는 gradient되지 않고 나머지 10~29개는 gradient update(backpropagation)됨

class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.
    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): 앵커의 면적입니다, 해당 면적은 :obj:`anchor_scales`의 요소 제곱과 참조 창의 원래 면적의 곱입니다.
    """

    # vgg16의 conv5 출력 feature map이 원본 이미지에 비해 16배 작다는 것을 의미한다?
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20, # object & background를 포함한 전체 수
                 ratios=[0.5, 1, 2], # Anchors의 너비와 높이의 비율
                 anchor_scales=[8, 16, 32] # Anchors의 면적
                 ):
                 
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios, anchor_scales=anchor_scales, feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )

class VGG16RoIHead(nn.Module):
    """
    VGG-16 기반 구현을 위한 Faster R-CNN Head.
    이 클래스는 Faster R-CNN의 헤드로 사용
    이것은 주어진 RoI의 피처 맵을 기반으로 클래스별 로컬라이제이션 및 분류를 출력

    인수:
    n_class(int): 배경을 포함할 수 있는 클래스 수.
    roi_size(int): RoI pooling 후 피처 맵의 높이와 너비.
    spatial_scale(float): roi의 스케일이 조정
    classifier(nn.Module): vgg16에서 포팅된 두 계층 선형
    """
    get_func_name = inspect.currentframe().f_code.co_name


    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.
        We assume that there are :math:`N` batches.
        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """
        # in case roi_indices is  ndarray

        if not hasattr(VGG16RoIHead, "_printed"):
            VGG16RoIHead._printed = False

        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois) #각 이미지 x에 맞게 roi pooling
        pool = pool.view(pool.size(0), -1) #flatten
        fc7 = self.classifier(pool) #fully connected
        roi_cls_locs = self.cls_loc(fc7) #regression
        roi_scores = self.score(fc7) #sofmax

        if not hasattr(VGG16RoIHead, '_printed'):
            VGG16RoIHead._printed = False
            debug_info = {
                "Function": get_func_name,
                "classifier": self.classifier,
                "self.cls_loc": self.cls_loc,
                "self.score": self.score,
                "self.n_class": self.n_class,
                "self.roi_size": self.roi_size,
                "self.spatial_scale":self.spatial_scale,
                "self.roi": self.roi,
                "n": n,
                "hh": hh,
                "ww": ww,
                "rpn_locs": rpn_locs,
                "rpn_scores": rpn_scores,
                "rpn_softmax_scores": rpn_softmax_scores,
                "rpn_fg_scores": rpn_fg_scores,
                "rois": rois,
                "indices_and_rois": indices_and_rois,
                "indices_and_rois": indices_and_rois,
                "pool": pool,
                "pool": pool,
                "fc7": fc7,
                "roi_cls_locs": roi_cls_locs,
                "roi_scores": roi_scores,
            }
            np.set_printoptions(precision=2, suppress=True)
            for key, value in debug_info.items():
                shape = value.shape if hasattr(value, 'shape') else 'N/A'
                if hasattr(value, 'shape'):
                    value_str = np.array2string(value, precision=2, suppress_small=True)
                else:
                    value_str = str(value)
                print(f"{key} (shape: {shape}):\n{value_str}\n")
            VGG16RoIHead._printed = True

        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
