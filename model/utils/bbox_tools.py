import numpy as np
import six
import sys

# Bounding box를 통해 Anchor를 proposal로 변환시키는 코드.
def loc2bbox(src_bbox, loc):
    """
    바운딩 박스 오프셋과 스케일에서 바운딩 박스를 디코딩한다.
    :meth:`bbox2loc`로 계산된 바운딩 박스 오프셋과 스케일이 주어지면,
    이 함수는 해당 정보를 2D 이미지 좌표 (y_min, x_min, y_max, x_max)로 디코딩.
    """

    if not hasattr(loc2bbox, "_printed"):
        loc2bbox._printed = False

    # 바운딩 박스가 없으면 빈 배열 반환
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    # 배열 복사 없이 데이터 타입 유지
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    # 각 바운딩 박스의 높이와 너비 계산
    src_height = src_bbox[:, 2] - src_bbox[:, 0]  # y_max - y_min
    src_width = src_bbox[:, 3] - src_bbox[:, 1]  # x_max - x_min

    # 각 바운딩 박스의 중심 좌표 계산 (중심 y, 중심 x)
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    # loc 배열은 각 바운딩 박스마다 4개의 값 (t_y, t_x, t_h, t_w)를 포함
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # 오프셋을 원본 높이와 너비에 곱하여 새로운 중심 좌표 계산
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]

    # 로그 스케일 변화량을 exp로 변환한 후 새로운 높이와 너비 계산
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    # 디코딩된 바운딩 박스 좌표를 저장할 배열 생성
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h  # y_min
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w  # x_min
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h  # y_max
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w  # x_max

    if not loc2bbox._printed:
        local_vars = locals()
        debug_info = {var: local_vars[var] for var in (
            "src_bbox", "src_height", "src_width", "src_ctr_y", "src_ctr_x",
            "dy", "dx", "dh", "dw", "ctr_y", "ctr_x", "h", "w", "dst_bbox"
        )}
        np.set_printoptions(precision=2, suppress=True)

        for key, value in debug_info.items():
            print(f"{key} (shape: {value.shape}):\n{np.array2string(value, precision=2, suppress_small=True)}\n")
        loc2bbox._printed = True

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
        소스 바운딩 박스와 대상 바운딩 박스를 "loc" 형식으로 인코딩한다.
       주어진 바운딩 박스를 이용하여 소스 바운딩 박스가 대상 바운딩 박스에 맞도록 이동(offset)과 스케일(scale)을 계산한다.

       수학적으로, 중심이 :math:`(y, x) = p_y, p_x`이고 크기가 :math:`p_h, p_w`인 바운딩 박스와,
       중심이 :math:`g_y, g_x`이고 크기가 :math:`g_h, g_w`인 대상 바운딩 박스가 주어지면,
       이동(offset)과 스케일(scale) 값 :math:`t_y, t_x, t_h, t_w`는 다음의 공식에 따라 계산된다.

           * :math:`t_y = \\frac{(g_y - p_y)}{p_h}`
           * :math:`t_x = \\frac{(g_x - p_x)}{p_w}`
           * :math:`t_h = \\log\\left(\\frac{g_h}{p_h}\\right)`
           * :math:`t_w = \\log\\left(\\frac{g_w}{p_w}\\right)`

       출력은 입력과 동일한 타입이다.
       이 인코딩 공식은 R-CNN [#]_과 같은 연구에서 사용된다.

       .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
              "Rich feature hierarchies for accurate object detection and semantic segmentation." CVPR 2014.

       Args:
           src_bbox (array): shape이 :math:`(R, 4)`인 이미지 좌표 배열이다.
               여기서 :math:`R`은 바운딩 박스의 개수를 의미하며, 해당 좌표들은
               :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}` 순서이다.
               -> 신경망을 통해서 나온 값(좌표이다)
           dst_bbox (array): shape이 :math:`(R, 4)`인 이미지 좌표 배열이다.
               해당 좌표들은 :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}` 순서이다.
               -> ground truth 좌표이다

       Returns:
           array:
               :obj:`src_bbox`에서 :obj:`dst_bbox`로의 바운딩 박스 이동(offset) 및 스케일(scale) 값들을 나타내는 배열이다.
               이 배열의 shape은 :math:`(R, 4)`이며, 두 번째 축은 각각 :math:`t_y, t_x, t_h, t_w` 값을 포함한다.
               -> 두 입력 좌표를 통해 계산하여 회귀(regression)로 예측할 값들이다.
    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    #bbox_a 1개와 bbox_b k개를 비교해야하므로 None을 이용해서 차원을 늘려서 연산한다.
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
        주어진 비율 및 스케일 열거에 의한 앵커 기본 창 생성
        - 주어진 비율에 따른 스케일 조정 및 변형 앵커 생성
        - 스케일 적용 앵커 면적의 주어진 비율 변형 보존
        - 생성 앵커 수: :obj:`R = len(ratios) * len(anchor_scales)`
        - :obj:`i * len(anchor_scales) + j` 번째 앵커: :obj:`ratios[i]`와 :obj:`anchor_scales[j]` 기반 앵커

        예시) 스케일 :math:`8`, 비율 :math:`0.25` 인 경우
        - 기준 창의 너비와 높이를 각각 :math:`8`배 확장
        - 앵커 변형 시 높이: 절반, 너비: 두 배

        [인자]
            base_size (number): 기준 창의 너비 및 높이
            ratios (list of floats): 앵커 너비와 높이 비율
            anchor_scales (list of numbers): 앵커 면적
                (면적 = 각 :obj:`anchor_scales` 요소 제곱 × 기준 창 원래 면적)

        [반환값]
            ~numpy.ndarray:
                형태: :math:`(R, 4)` 배열
                각 요소: 바운딩 박스 좌표 집합 (순서: :math:`(y_{min}, x_{min}, y_{max}, x_{max})`)
    """
    # debug_utils
    debug_info={}
    if not hasattr(generate_anchor_base, "_printed"):
        generate_anchor_base._printed=False

    py = base_size / 2. # py= 16 / 2 -> 8.0
    px = base_size / 2. # px= 16 / 2 -> 8.0

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32) # anchor_base: shape(9, 4)
    for i in six.moves.range(len(ratios)): # len(ratios) - > int:3 # i: 0 0 0 1 1 1 2 2 2
        for j in six.moves.range(len(anchor_scales)): #  len(ratios) - > int:3 # j: 0 1 2 0 1 2 0 1 2
            print(f"i: {i}, j: {j}") # i:0, j:0

            # np.sqrt: 입력된 값의 제곱근을 계산 -> 주어진 배열, 숫자에 대해 각 원소의 제곱근 구하고 동일한 모양의 배열로 반환
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i]) # h: 90.51
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i]) # w: 181.02
            index = i * len(anchor_scales) + j # index: 0
            # i: 0, j: 1
            # i: 0, j: 2
            # i: 1, j: 0
            # i: 1, j: 1
            # i: 1, j: 2
            # i: 2, j: 0
            # i: 2, j: 1
            # i: 2, j: 2
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

            # printing to debug_info output
            if not generate_anchor_base._printed:
                output_vals = locals()
                debug_info = {var: output_vals[var] for var in
                              ("py", "anchor_base", "h", "w", "index", "i", "j")}

                np.set_printoptions(precision=2, suppress=True)

                for key, value in debug_info.items():
                    shape = value.shape if hasattr(value, 'shape') else 'N/A'
                    if hasattr(value, 'shape'):
                        value_str = np.array2string(value, precision=2, suppress_small=True)
                    else:
                        value_str = str(value)
                    print(f"{key} (shape: {shape}):\n{value_str}\n")
                generate_anchor_base._printed = True

    return anchor_base

if __name__ == '__main__':
    pass
# python train.py train --env='fasterrcnn' --plot-every=100
