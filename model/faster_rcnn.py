from __future__ import absolute_import
from __future__ import division
import torch
import numpy as np
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from torchvision.ops import nms

import torch.nn as nn
from VOCdevkit.dataset import preprocess
import torch.nn.functional as F
from utils.config import opt

def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):
    """
        Faster R-CNN 기본 클래스
        - 객체 검출 API 지원 Faster R-CNN 링크 구현
        - 구성 요소
            • Feature extraction: 입력 이미지 → feature map 산출
            • Region Proposal Networks: feature map 활용 → 객체 주변 RoIs 집합 생성
            • Localization and Classification Heads: 제안된 RoIs의 feature map 활용 → RoI 내 객체 카테고리 분류 및 localization 개선
        - 각 단계 담당: 호출 가능한 :class:`torch.nn.Module` 객체 (예: feature, rpn, head)
        - 제공 함수
            • :meth:`predict`: 입력 이미지 → 이미지 좌표 bounding box 산출 (블랙박스 방식 활용)
            • :meth:`__call__`: 학습 및 디버깅 시 중간 출력 제공
        - 기타: 객체 검출 API 지원 다른 링크들도 동일 인터페이스 :meth:`predict` 구현 (자세한 내용은 :meth:`predict` 참조)

        [인자]
            extractor (nn.Module): BCHW 이미지 배열 입력, feature map 산출 모듈
            rpn (nn.Module): :class:`model.region_proposal_network.RegionProposalNetwork`와 동일 인터페이스 구현 모듈
            head (nn.Module): BCHW 변수, RoIs 및 RoI에 대응하는 배치 인덱스 입력, 클래스별 localization 파라미터 및 클래스 점수 산출 모듈
            loc_normalize_mean (tuple of four floats): localization 추정치 평균값
            loc_normalize_std (tuple of four floats): localization 추정치 표준편차
    """


    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    # Total number of classes including the background.
    @property # 외부에서 class 속성을 내부 속성처럼 접근 가능함.
    def n_class(self):
        return self.head.n_class

    def get_n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        """
        Faster R-CNN의 순전파(Forward)를 수행한다.
        RPN은 입력 이미지에 적용된 스케일링 파라미터 :obj:`scale`을 활용해,
        작은 객체를 선택할 임계값을 결정하며, 이 임계값에 미달하는 객체들은 신뢰도와 관계없이 배제한다.
        다음 기호들을 사용한다:
            * :math:`N`은 배치의 크기를 나타낸다.
            * :math:`R'`은 배치 전체에서 생성된 총 RoI 수를 나타낸다.
              각 이미지의 제안된 RoI 수를 :math:`R_i`라 할 때,
              :math:`R' = \\sum_{i=1}^{N} R_i`이다.
            * :math:`L`은 배경을 제외한 클래스의 수를 나타낸다.
              클래스들은 배경, 첫 번째 클래스, …, 그리고 :math:`L`번째 클래스 순으로 정렬된다.

        Args:
            x (autograd.Variable): 4D 이미지 변수이다.
            scale (float): 전처리 과정에서 원본 이미지에 적용한 스케일링 비율이다.

        Returns:
            Variable, Variable, array, array:
            다음 네 가지 값을 튜플로 산출한다.
                * **roi_cls_locs**: 제안된 RoI에 대해 오프셋과 스케일링 정보를 산출하며,
                  그 shape는 :math:`(R', (L+1) \\times 4)`이다.
                * **roi_scores**: 제안된 RoI에 대한 클래스 예측 결과를 산출하며,
                  그 shape는 :math:`(R', L+1)`이다.
                * **rois**: RPN이 제안한 RoI들을 산출하며, 그 shape는 :math:`(R', 4)`이다.
                * **roi_indices**: 각 RoI에 대응하는 배치 인덱스를 산출하며, 그 shape는 :math:`(R',)`이다.
        """

        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices #RoI loss(fast R-CNN loss)를 구할때 사용되는 값들 -> 즉 해당 class는 fast R-CNN의 출력을 구하기 위한 class

    def use_preset(self, preset):
        """Use the given preset during prediction.
        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.
        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.
        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.
        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob): #
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs,sizes=None,visualize=False):
        """Detect objects from images.
        This method predicts objects for each image.
        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.
        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.
           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.
        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(at.totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
