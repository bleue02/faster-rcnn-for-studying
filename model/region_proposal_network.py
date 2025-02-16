import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
import inspect
from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
    """
        Faster R-CNN에 도입된 Region Proposal Network.
        - Faster R-CNN [#]_에 소개된 Region Proposal Network로,
          입력 이미지로부터 추출한 feature를 기반으로 객체 주변의 클래스에 무관한 bounding box 제안을 수행

        참고:
        .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
               Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.

        [인자]
            in_channels (int): 입력 채널 수이다.
            mid_channels (int): 중간 텐서 채널 수
            ratios (list of floats): 앵커의 너비와 높이 비율
            anchor_scales (list of numbers): 앵커의 면적이다. 각 면적은 :obj:`anchor_scales` 요소 제곱과 참조 창의 원래 면적 곱으로 계산
            feat_stride (int): 입력 feature 추출 후 적용되는 스트라이드 크기
            initialW (callable): 초기 가중치 값이다. :obj:`None`일 경우 0.1 스케일의 Gaussian 분포를 이용해 가중치를 초기화하며,
                                 배열을 입력받아 값을 수정하는 callable로도 사용
            proposal_creator_params (dict): :class:`model.utils.creator_tools.ProposalCreator`에 사용되는 키-값 파라미터

        [참고 사항]
            :class:`~model.utils.creator_tools.ProposalCreator`
    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict(),
    ): # proposal_Creator_params=dict() --> dit(): parameter없이 호출 즉 empty dictionary로 initialized

        super(RegionProposalNetwork, self).__init__()
        # self.anchor_base:
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios) # anchor에서 9개의 anchor box 생성 --> Anchor(9, 4)
        self.feat_stride = feat_stride # feat_stride->16

        # self.proposal_layer: nms제거 후
        # 여기서 self는 RPN이 되고 proposalcreator의 인자로 들어가서 해당 네트워크가 "training인지 testing인지 알려준다."
        # self.proposal_layaer: (shape: (2000, 4)) --> [ymin, xmin, ymax, xmax]로 이루어진 2000개의 바운딩 박스
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0] # n_anchor:  Anchor(9, 4) --> n_anchor: 9
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """
        Region Proposal Network 전방 전달.
        """
        get_func_name = inspect.currentframe().f_code.co_name
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), # self.anchor_base->(9,4) -> get[0] --> '9'
            self.feat_stride, hh, ww)  # hh, ww 기준으로 conv -> feature map -> draw grid on image # 눈으로 확인할 필요 있음.

        debug_info = {}
        if not hasattr(self.__class__, '_printed') or not self.__class__._printed:
            print('AANCHHOR: ', anchor)
            print('AANCHHOR_type:', type(anchor))
            print('AANCHHOR_shape: ', anchor.shape)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        if not hasattr(RegionProposalNetwork, '_printed'):
            RegionProposalNetwork._printed = False
            debug_info = {
                "Function": get_func_name,
                "self.anchor_base": self.anchor_base,
                "feat_stride": self.feat_stride,
                "n_anchor": n_anchor,
                "conv1": self.conv1,
                "score": self.score,
                "loc": self.loc,
                "h (after ReLU conv1)": h,
                "n": n,
                "hh": hh,
                "ww": ww,
                "rpn_locs": rpn_locs,
                "rpn_scores": rpn_scores,
                "rpn_softmax_scores": rpn_softmax_scores,
                "rpn_fg_scores": rpn_fg_scores,
                "rois": rois,
                "roi_indices": roi_indices,
                "anchor": anchor,
                "batch_index": batch_index
            }
            np.set_printoptions(precision=2, suppress=True)
            for key, value in debug_info.items():
                # 만약 value가 torch.Tensor라면, CPU로 옮긴 후 넘파이 배열로 변환
                if isinstance(value, torch.Tensor):
                    value_np = value.detach().cpu().numpy()
                elif hasattr(value, 'shape'):
                    value_np = value  # 이미 numpy array일 경우
                else:
                    value_np = value
                shape = value_np.shape if hasattr(value_np, 'shape') else 'N/A'
                if hasattr(value_np, 'shape'):
                    value_str = np.array2string(value_np, precision=2, suppress_small=True)
                else:
                    value_str = str(value_np)
                print(f"{key} (shape: {shape}):\n{value_str}\n")
            RegionProposalNetwork._printed = True

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # ***그냥 50x50의 feature map에 22500개의 여러 크기를 가진 anchor box가 image에 덮혀져 있다고 생각하면 됨.

    # Enumerate all shifted anchors:
    # anchor_base는 하나의 anchor에 9개 종류의 anchor 갖고, 이를 enumerate시켜 전체 이미지에 각각의 anchor를 갖게 한다
    #즉 50x50의 feature map에서 각각의 pixel은 하나의 anchor를 나타내므로 50x50x9개의 anchor box의 좌표를 구해준다.
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    debug_info={}
    if not hasattr(_enumerate_shifted_anchor,'_printed'):
        _enumerate_shifted_anchor._printed=False

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # np.meshgrid: 1차원 array를 통해 x, y를 가지고 data에 gride생성.
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), # ravel: 2차원이면 1차원으로 펼침
                      shift_y.ravel(), shift_x.ravel()), axis=1) # stack x, y를 열방향으로 stack

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    if not _enumerate_shifted_anchor._printed:
        output_vals = locals()
        debug_info = {var: output_vals[var] for var in
                      ("shift_y", "shift_x", "shift_x", "shift_y", "shift", "A", "K", "anchor")}

        np.set_printoptions(precision=2, suppress=True)

        for key, value in debug_info.items():
            shape = value.shape if hasattr(value, 'shape') else 'N/A'
            if hasattr(value, 'shape'):
                value_str = np.array2string(value, precision=2, suppress_small=True)
            else:
                value_str = str(value)
            print(f"{key} (shape: {shape}):\n{value_str}\n")
        _enumerate_shifted_anchor._printed = True

    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter3
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
