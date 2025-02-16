import numpy as np
import matplotlib.pyplot as plt
import cv2


# _enumerate_shifted_anchor 함수에서 사용된 shift 계산 부분을 분리해서 실행해보기
def visualize_shift(feat_stride, height, width):
    # 50x50 feature map이라면:
    # height, width = 50, 50
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    print(shift_y)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    print(shift_x)
    # meshgrid: x, y 좌표의 그리드 생성
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    print(shift_y)
    print(shift_x)

    # shift 배열 생성: 각 픽셀 위치마다 [y, x, y, x] 형식의 오프셋을 만듦
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    return shift, shift_x, shift_y


# 예시 파라미터 (feat_stride=16, feature map size: 50x50)
feat_stride = 16
height = 50
width = 50
shift, shift_x, shift_y = visualize_shift(feat_stride, height, width)

# 이미지 불러오기 (예시 경로)
image_path = r"C:\Users\jdah5454\Desktop\lena짱.png"
img = cv2.imread(image_path)
if img is not None:
    # OpenCV는 BGR이므로 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
else:
    # 이미지가 없으면 빈 캔버스 생성 (예시)
    img = np.ones((height * feat_stride, width * feat_stride, 3), dtype=np.uint8) * 255

# 시각화: 이미지 위에 shift 좌표(즉, 각 anchor의 기준점)를 scatter plot으로 표시
plt.figure(figsize=(8, 8))
plt.imshow(img)
# shift_x, shift_y는 2차원 배열이므로, ravel()해서 1차원으로 만듭니다.
plt.scatter(shift_x.ravel(), shift_y.ravel(), marker='x', color='red', s=10)
plt.axis('off')
plt.show()
