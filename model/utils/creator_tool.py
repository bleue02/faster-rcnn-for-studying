import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


class ProposalTargetCreator:
    """roi(nms를 거친 roi)에 해당되는 ground truth와 iou를 비교하여 먼저 positive/negative를 sampling한다(논문 기준 128개) -> sample_roi

    해당 객체 클래스에 맞게 labeling하고(배경:0, 클래스1:1, ..., 클래스n:n) sample_roi의 인덱스와 맞는 label들만 사용한다.
    sample roi와 gt_bbox를 이용해 bbox regression에서 regression해야할 ground truth loc값 t_x, t_y, t_w, t_h 을 구한다.
    따라서 return으로 sample_roi, ground truth loc와 label이다.
    이를 이용해 fast R-CNN loss(RoI loss)를 계산할 때, sample_roi는 네트워크 입력으로, loc와 label은 gt로 사용한다.
    Assign ground truth bounding boxes to given RoIs.
    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.
    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.
    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.
        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.
        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.
        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.
        Returns:
            (array, array, array):
            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.
        """
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """Anchor에 해당되는 ground truth와 iou를 비교하여 positive, negative, ignore sample을 labeling하고
    bbox regression에서 regression해야할 ground truth loc값 t_x, t_y, t_w, t_h 을 구한다.
    따라서 return으로 ground truth loc와 label이다.
    이를 이용해 RPN loss를 계산할 때, gt로 사용한다.
    Assign the ground truth bounding boxes to anchors.
    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.
    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.
    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.
    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.
        Types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.
        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.
        Returns:
            (array, array):
            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.
        """

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (VOCdevkit) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


import numpy as np
import torch
import sys

class ProposalCreator:
    """
    Feature map → (conv3x3, regression conv) 적용 → loc (bounding box 위치 오프셋 [dx, dy, dw, dh]) 산출
    loc와 anchor를 입력받아 bbox 형태의 좌표 (ROI)로 변환.
    조건에 부합하는 ROI만 선별한 후, Non-Maximum Suppression (NMS)을 적용하여 지정된 개수의 ROI를 반환함.
    Proposal regions (객체 검출 제안)을 산출하는 객체.

    이 객체의 :meth:`__call__` 메서드는 예측된 bounding box offset을 anchors에 적용하여
    객체 검출 제안을 생성함. NMS 전후 유지할 bbox 개수를 제어하는 파라미터를 입력받으며,
    음수의 경우 입력된 모든 bbox 또는 NMS 반환 bbox를 사용함.
    이 클래스는 Faster R-CNN [#]_에 도입된 Region Proposal Networks에 활용됨.
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
       Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): NMS 적용 시 사용할 임계값.
        n_train_pre_nms (int): 학습 모드에서 NMS 전 유지할 최고 점수 bbox 개수.
        n_train_post_nms (int): 학습 모드에서 NMS 후 유지할 최고 점수 bbox 개수.
        n_test_pre_nms (int): 테스트 모드에서 NMS 전 유지할 최고 점수 bbox 개수.
        n_test_post_nms (int): 테스트 모드에서 NMS 후 유지할 최고 점수 bbox 개수.
        force_cpu_nms (bool): True이면 항상 CPU 모드 NMS를 사용. (False이면 입력 타입에 따라 결정)
        min_size (int): bbox 크기 기준 임계값.
    """
    _printed = False

    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=6000, n_test_post_nms=300, min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """
        입력 ndarray를 사용하여 Region Proposal을 산출함.

        Args:
            loc (array): 앵커에 대한 예측 위치 오프셋 및 스케일 값, shape: (R, 4).
            score (array): 각 앵커의 예측 전경(Foreground) 확률, shape: (R,).
            anchor (array): 앵커 박스 좌표, shape: (R, 4).
            img_size (tuple of ints): 크기 조정 후 이미지의 (높이, 너비).
            scale (float): 이미지 파일 읽기 후 적용된 스케일 비율.

        Returns:
            array: NMS 적용 후 선별된 Proposal bbox 좌표, shape: (S, 4).
        """
        debug_info = {}

        # 학습/테스트 모드에 따른 NMS 전후 유지할 bbox 개수 설정
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms  # 학습 시: NMS 전 12,000개
            n_post_nms = self.n_train_post_nms  # 학습 시: NMS 후 2,000개
        else:
            n_pre_nms = self.n_test_pre_nms   # 테스트 시: NMS 전 6,000개
            n_post_nms = self.n_test_post_nms  # 테스트 시: NMS 후 300개

        # 앵커에 bbox 변환 오프셋 적용 → ROI 산출
        roi = loc2bbox(anchor, loc)
        debug_info["ROI (초기)"] = roi.copy() # (shape: (16650, 4))

        # 예측 bbox 좌표를 이미지 경계 내로 클립
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # bbox 높이 및 너비 산출 (높이: y_max - y_min, 너비: x_max - x_min)
        hs = roi[:, 2] - roi[:, 0] # (hs) (shape: (16650,)):
        ws = roi[:, 3] - roi[:, 1] # (ws) (shape: (16650,)):
        debug_info["높이 (hs)"] = hs.copy()
        debug_info["너비 (ws)"] = ws.copy()

        # 임계 크기 이상 bbox 인덱스 선택
        min_size_val = self.min_size * scale
        keep_size = np.where((hs >= min_size_val) & (ws >= min_size_val))[0]
        debug_info["크기 필터 인덱스"] = keep_size.copy()

        roi = roi[keep_size, :]
        score = score[keep_size]
        debug_info["ROI (크기 필터 후)"] = roi.copy() # (shape: (16650,)):
        debug_info["score (크기 필터 후)"] = score.copy() # (shape: (16650, 4)):

        # 점수 내림차순 정렬 (최고 점수 우선)
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        debug_info["정렬 인덱스 (pre-NMS)"] = order.copy() # (pre-NMS) (shape: (12000,)):
        roi = roi[order, :]
        score = score[order]
        debug_info["ROI (정렬 후)"] = roi.copy() # (shape: (12000, 4)):
        debug_info["score (정렬 후)"] = score.copy() # (shape: (12000,))

        # NMS 적용: torch 텐서로 변환 후 NMS 수행 (GPU 사용)
        keep_nms = nms(torch.from_numpy(roi).cuda(),
                       torch.from_numpy(score).cuda(),
                       self.nms_thresh)
        if n_post_nms > 0:
            keep_nms = keep_nms[:n_post_nms]
        debug_info["NMS 인덱스"] = keep_nms.cpu().numpy().copy() # (shape: (2000,))

        roi = roi[keep_nms.cpu().numpy()]
        debug_info["ROI (NMS 후)"] = roi.copy() #  (shape: (2000, 4))

        if not ProposalCreator._printed:
            np.set_printoptions(precision=2, suppress=True)
            for key, value in debug_info.items():
                if isinstance(value, np.ndarray):
                    print(f"{key} (shape: {value.shape}):\n{np.array2string(value, precision=2, suppress_small=True)}\n")
                else:
                    print(f"{key}: {value}\n")
            ProposalCreator._printed = True

        return roi



