# ===============================================================================
# Name      : transform.py
# Version   : 1.0.0
# Brief     :
# Time-stamp: 2024-01-10 10:34
# Copyirght 2022 Hiroya Aoyama
# ===============================================================================

import numpy as np
import cv2
from pydantic import BaseModel
from typing import Union, Optional, Tuple
try:
    from logger import setup_logger
    logger = setup_logger(__name__)
except Exception:
    from logging import getLogger
    logger = getLogger(__name__)


class PosData(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0
    r: float = 0


def calc_offset_vector(u: Union[list, np.ndarray], angle: float) -> np.ndarray:
    """検出点から回転中心までのオフセットベクトルを計算

    Args:
        u (Union[list, np.ndarray]): _description_
        angle (float): 単位degree, ↓+ ↑-

    Returns:
        np.ndarray: _description_
    """
    angle = np.deg2rad(angle)
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    return np.dot(matrix, np.array(u))


def trans_img2robot_coordinates(x: float, y: float, matrix: Union[list, np.ndarray],
                                *, ignore_translation: bool = False) -> Tuple[float, float]:
    """画像座標からロボット座標に変換する\n
    並進成分を使う場合はuse_t_componentをTrueに変更
    """
    if ignore_translation:
        # NOTE: matrixの並進成分初期化
        matrix[0][2] = 0.0
        matrix[1][2] = 0.0
    ans = np.dot(np.linalg.inv(matrix), np.array([x, y, 1]))  # type: ignore
    return ans[0], ans[1]


def calc_resolution(matrix: list) -> float:
    """キャリブレーションデータから分解能を計算"""
    xx, xy = matrix[0][0], matrix[0][1]
    yx, yy = matrix[1][0], matrix[1][1]
    if xx == 0 and xy == 0:
        return 0.0
    if yx == 0 and yy == 0:
        return 0.0

    x_resol = 1 / np.sqrt(xx**2 + xy**2)
    y_resol = 1 / np.sqrt(yx**2 + yy**2)

    return round((x_resol + y_resol) / 2, 4)


def calc_grid(left_top: Union[list, tuple], left_bottom: Union[list, tuple],
              right_top: Union[list, tuple], right_bottom: Union[list, tuple],
              num_row: int, num_col: int,
              *,
              left_margin: int = 0, right_margin: int = 0,
              top_margin: int = 0, bottom_margin: int = 0,
              to_integer: bool = False, decimal: int = 2) -> list:
    """4点からグリッドを生成"""
    pts = []
    top_left = np.array(left_top) + np.array([left_margin, top_margin])
    top_right = np.array(right_top) + np.array([-right_margin, top_margin])
    bottom_left = np.array(left_bottom) + np.array([left_margin, -bottom_margin])
    bottom_right = np.array(right_bottom) + np.array([-right_margin, -bottom_margin])

    left_points = np.linspace(top_left, bottom_left, num_row)
    right_points = np.linspace(top_right, bottom_right, num_row)

    for pt in zip(left_points, right_points):
        h_points = np.linspace(pt[0], pt[1], num_col)
        for h_point in h_points:
            x = h_point[0]
            y = h_point[1]
            if to_integer:
                pts.append([round(x), round(y)])
            else:
                pts.append([round(x, decimal), round(y, decimal)])

    return pts


def calc_robot_axis(matrix: list) -> Tuple[list, list]:
    """マトリックスからロボットの軸座標ベクトルを求める"""
    m = matrix.copy()
    m[0][2] = 0.0
    m[1][2] = 0.0
    mat = np.array(m)
    # NOTE: 軸座標ベクトルを計算
    x_vec = np.dot(mat, np.array([1, 0, 1]))
    y_vec = np.dot(mat, np.array([0, 1, 1]))
    # NOTE: ノルムを計算
    x_norm = np.linalg.norm(x_vec[:2], ord=2)  # type: ignore
    y_norm = np.linalg.norm(y_vec[:2], ord=2)  # type: ignore
    # NOTE: 単位ベクトル化
    x_vec = x_vec[:2] / x_norm
    y_vec = y_vec[:2] / y_norm
    return x_vec, y_vec


def calc_vector_angle(u: np.ndarray) -> float:
    """ベクトルの角度を求める"""
    return np.rad2deg(np.arctan2(u[1], u[0]))


def angle_converter(angle: float) -> float:
    """±180度を0-360度に変換"""
    if angle < 0:
        return 360.0 - angle
    return angle


def get_rotation_direction(x_vec: list, y_vec: list) -> bool:
    """回転方向がロボット座標と一致していればTrue,逆であればFalse"""
    x_ang = calc_vector_angle(np.array(x_vec))
    y_ang = calc_vector_angle(np.array(y_vec))
    x_ang = angle_converter(x_ang)
    y_ang = angle_converter(y_ang)
    diff = x_ang - y_ang
    if abs(diff) > 180:
        diff = -diff
    if diff > 0:
        return True
    return False

# ===============================================================================
# NOTE: 計算式
# ===============================================================================


def calc_distance_from_rot_center(x: float, y: float, cx: float, cy: float) -> Tuple[float, float]:
    dx = x - cx
    dy = y - cy
    return dx, dy


def output_xyr(pos: PosData,
               b_pos: PosData,
               matrix: list,
               rot_c: PosData,
               *,
               ignore_translation: bool = False,
               ) -> Tuple[list, list, list]:
    """多分他で使っていないとは思うが返り値フォーマットを変えるとバグ起きるので既存のこの関数を残しておく"""

    # NOTE: 回転中心からの距離
    dx1, dy1 = calc_distance_from_rot_center(b_pos.x, b_pos.y,
                                             rot_c.x, rot_c.y)  # NOTE: ベース位置
    dx2, dy2 = calc_distance_from_rot_center(pos.x, pos.y,
                                             rot_c.x, rot_c.y)  # NOTE: 検出位置

    # NOTE: ロボット座標に変換
    x1_rb, y1_rb = trans_img2robot_coordinates(dx1, dy1, matrix,
                                               ignore_translation=ignore_translation)  # NOTE: ベース位置
    x2_rb, y2_rb = trans_img2robot_coordinates(dx2, dy2, matrix,
                                               ignore_translation=ignore_translation)  # NOTE: 検出位置

    return [x1_rb, y1_rb], [x2_rb, y2_rb], [b_pos.r, pos.r]


# ===============================================================================
# NOTE: 描画系
# ===============================================================================


def draw_estimated_result(src: np.ndarray,
                          pts: Union[list, np.ndarray],
                          circle_: Optional[tuple] = None) -> np.ndarray:
    """取得点の描画"""

    thickness = 7
    color = (10, 200, 10)
    dst = src.copy()

    if circle_ is not None:
        dst = cv2.circle(
            img=dst,
            center=(int(circle_[0]), int(circle_[1])),
            radius=int(circle_[2]),
            color=(200, 10, 10),
            thickness=thickness
        )

    points_ = np.round(pts).astype(np.int32)  # type: ignore
    for pt in points_:
        dst = cv2.circle(
            img=dst,
            center=(pt[0], pt[1]),
            radius=thickness,
            color=color,
            thickness=-1
        )

    return dst


def visualize_robot_axis(dst: np.ndarray, matrix: list, pt: list) -> np.ndarray:
    """ロボットの軸を画像に可視化"""
    px, py = pt
    dist = 50
    x_uvec, y_uvec = calc_robot_axis(matrix)
    x_vec = (int(x_uvec[0] * dist) + px, int(x_uvec[1] * dist) + py)
    y_vec = (int(y_uvec[0] * dist) + px, int(y_uvec[1] * dist) + py)

    dst = cv2.arrowedLine(dst,
                          pt1=(px, py),
                          pt2=x_vec,
                          color=(0, 255, 0),
                          thickness=3,
                          line_type=cv2.LINE_4,
                          shift=0,
                          tipLength=0.1)

    dst = cv2.arrowedLine(dst,
                          pt1=(px, py),
                          pt2=y_vec,
                          color=(0, 255, 0),
                          thickness=3,
                          line_type=cv2.LINE_4,
                          shift=0,
                          tipLength=0.1)

    dst = cv2.putText(dst,
                      text='X',
                      org=x_vec,
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=1.0,
                      color=(0, 255, 0),
                      thickness=2,
                      lineType=cv2.LINE_4)

    dst = cv2.putText(dst,
                      text='Y',
                      org=y_vec,
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=1.0,
                      color=(0, 255, 0),
                      thickness=2,
                      lineType=cv2.LINE_4)

    return dst


# def get_nearest_points(point: list, calib_points: list, require: int = 4) -> list:
#     """暫定"""

#     calib_np = np.array(calib_points)
#     pos_np = np.array(point)

#     dist = []
#     for calib in calib_np:
#         u = pos_np - calib
#         dist.append(np.linalg.norm(u))  # type: ignore

#     distance = np.array(dist)  # type: ignore

#     descend_index = distance.argsort()

#     if len(descend_index) > require:
#         # indexs = descend_index[:require - 1].copy()
#         while True:
#             try:
#                 indexs = descend_index[:require].copy()
#             except Exception:
#                 return [-1]
#             x_list = []
#             y_list = []
#             for i, j in enumerate(indexs):
#                 x_list.append(calib_points[j][0])
#                 y_list.append(calib_points[j][1])
#             x_diff = np.var(x_list)
#             y_diff = np.var(y_list)

#             if x_diff > y_diff / 10 and y_diff > x_diff / 10:
#                 break

#             descend_index = np.delete(descend_index, 2)

#         return indexs
#     elif len(descend_index) >= 4:
#         return descend_index[:4]
#     else:
#         return [-1]


if __name__ == '__main__':
    pass
