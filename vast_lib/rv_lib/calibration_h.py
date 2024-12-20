# ===============================================================================
# Name      : transform.py
# Version   : 1.0.0
# Brief     :
# Time-stamp: 2023-03-27 12:39
# Copyirght 2022 Hiroya Aoyama
# ===============================================================================

import numpy as np
import cv2
import math
from pydantic import BaseModel
from typing import Union, Optional
try:
    from logger import setup_logger
    logger = setup_logger(__name__)
except Exception:
    from logging import getLogger
    logger = getLogger(__name__)


class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class PosData(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0
    r: float = 0


class EstimatedCircle(BaseModel):
    x: float
    y: float
    r: float
    rmse: float


class EstimatedMatrix(CustomBaseModel):
    matrix: list
    x_rmse: float
    y_rmse: float


def mean_squared_error(y_true: Union[np.ndarray, list],
                       y_pred: Union[np.ndarray, list],
                       sample_weight: Optional[np.ndarray] = None,
                       ) -> float:
    """Root平均二乗誤差の計算"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    output_errors = np.average((y_true - y_pred) ** 2, axis=0,
                               weights=sample_weight)
    # NOTE Rootをかける
    output_errors = np.sqrt(output_errors)

    return np.average(output_errors)


# ===============================================================================
# NOTE: 計算式
# ===============================================================================

def circle_estimation(*, pts: Optional[Union[list, np.ndarray]] = None,
                      x_data: Optional[Union[list, np.ndarray]] = None,
                      y_data: Optional[Union[list, np.ndarray]] = None) -> Optional[EstimatedCircle]:
    """円の中心を計算
    入力はlistでもnp.ndarrayでも可\n
    [[x1,y1], [x2,y2], ... ,[xn,yn]]の形式で入力する場合はptsに\n
    x=[x1,x2,...,xn] y=[y1,y2,...,yn]の形式で入力する場合はx_data, y_dataに入力

    Args:
        pts (Union[list, np.ndarray], optional): _description_. Defaults to None.
        x_data (Union[list, np.ndarray], optional): _description_. Defaults to None.
        y_data (Union[list, np.ndarray], optional): _description_. Defaults to None.

    Returns:
        Optional[EstimatedCircle]: _description_
    """
    # NOTE: 計算式はここを参照
    # http://imagingsolution.blog107.fc2.com/blog-entry-16.html

    # NOTE: [[x1,y1], [x2,y2], ... ,[xn,yn]]の形式で入力
    if pts is not None:
        pts = np.array(pts)
        x = pts.T[0]
        y = pts.T[1]

    # NOTE: x=[x1,x2,...,xn] y=[y1,y2,...,yn]の形式で入力
    elif (x_data is not None) and (y_data is not None):
        x = np.array(x_data).astype(np.float32)
        y = np.array(y_data).astype(np.float32)
        if x.size != y.size:
            logger.error('Input Dimention is not match')
            return None
    else:
        # NOTE: 引数がないとき
        logger.error('Arguments is None')
        return None

    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum([ix ** 2 for ix in x])
    sum_y2 = sum([iy ** 2 for iy in y])
    sum_xy = sum([ix * iy for (ix, iy) in zip(x, y)])

    a_mat = np.array([
        [sum_x2, sum_xy, sum_x],
        [sum_xy, sum_y2, sum_y],
        [sum_x, sum_y, len(x)]
    ])

    b_mat = np.array([
        [-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
        [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
        [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]
    ])

    try:
        ans = np.linalg.inv(a_mat).dot(b_mat)  # type: ignore
    except Exception as e:
        # NOTE: 円近似できない場合
        logger.error(f'Calculation Error\n {e}')
        return None

    cx = float(ans[0] / -2)
    cy = float(ans[1] / -2)
    radius = math.sqrt(cx**2 + cy**2 - ans[2])

    # NOTE: Rで近似結果を評価
    r_predict = [radius] * len(x)
    r_measured = [math.sqrt((ix - cx)**2 + (iy - cy)**2) for (ix, iy) in zip(x, y)]

    rmse_r = mean_squared_error(r_measured, r_predict)

    return EstimatedCircle(x=round(cx, 3),
                           y=round(cy, 3),
                           r=round(radius, 3),
                           rmse=round(rmse_r, 3))


def calibration_with_2d_points(robot_points: Union[list, np.ndarray],
                               image_points: Union[list, np.ndarray]) -> Optional[EstimatedMatrix]:
    """キャリブレーションデータを計算

    Args:
        rbt_pts (Union[list, np.ndarray]): _description_
        img_pts (Union[list, np.ndarray]): _description_

    Returns:
        Optional[EstimatedMatrix]: _description_
    """

    robot_points = np.array(robot_points).T
    image_points = np.array(image_points).T

    # rbt_pts = [x1,x2,...,xn], [y1,y2,...,yn]
    # img_pts = [x1,x2,...,xn], [y1,y2,...,yn]

    num_pts: int = len(robot_points[0])

    x_mat = np.array([robot_points[0],
                      robot_points[1],
                      np.ones(num_pts)])

    # NOTE: 行列で一括計算する方法（疑似逆行列）
    # y_mat = np.array([img_pts[0],
    #                   img_pts[1],
    #                   np.ones(num_pts)])
    #
    # mat = y_mat @ x_mat @ np.linalg.inv(np.dot(x_mat, x_mat.T))
    # mat = np.dot(np.dot(y_mat, x_mat.T), np.linalg.inv(np.dot(x_mat, x_mat.T))).tolist()

    # NOTE: 方程式で解く方法
    try:
        m_x = np.linalg.solve(x_mat.dot(x_mat.T), x_mat.dot(image_points[0]))  # type: ignore
        m_y = np.linalg.solve(x_mat.dot(x_mat.T), x_mat.dot(image_points[1]))  # type: ignore
    except Exception as e:
        # TODO: pinvで求める
        # https://qiita.com/masayas/items/1c393460736e3fb71a80
        logger.error(f'\n {e}')
        return None

    # NOTE: 算出した行列式から推定値を算出
    # NOTE: ロボット座標から画像座標に変換
    x_predict = np.dot(m_x, x_mat)
    y_predict = np.dot(m_y, x_mat)

    # NOTE: 入力と出力の誤差RMSE（画像座標で比較）
    rmse_x = mean_squared_error(image_points[0], x_predict)
    rmse_y = mean_squared_error(image_points[1], y_predict)

    # NOTE: numpyからlistに変換して少数点第7位で切り捨て
    ans = np.array([m_x, m_y, [0, 0, 1]]).tolist()
    ans = [[round(ans[i][j], 7) for j in range(0, 3)] for i in range(0, 3)]

    return EstimatedMatrix(matrix=ans,
                           x_rmse=round(rmse_x, 3),
                           y_rmse=round(rmse_y, 3))


def relative_calibration(teaching_robot_point: Union[list, tuple],
                         teaching_image_point: Union[list, tuple],
                         robot_points: Union[list, np.ndarray],
                         image_points: Union[list, np.ndarray],
                         shift_amount: tuple = (0.0, 0.0)) -> Optional[EstimatedMatrix]:
    """相対位置のキャリブレーション

    Args:
        teaching_robot_point (Union[list, tuple]): ティーチング座標
        teaching_image_point (Union[list, tuple]): ティーチング位置での画像座標
        robot_points (Union[list, np.ndarray]): キャリブレーションポイント(ロボット座標)
        image_points (Union[list, np.ndarray]): キャリブレーションポイント(画像座標)
        shift_amount (tuple, optional): キャリブレーションワークと実ワークの画像座標での差分

    Returns:
        _type_: _description_
    """

    # NOTE: 相対位置座標
    relative_robot_positions = np.array(robot_points) - np.array(teaching_robot_point)
    relative_image_positions = np.array(image_points) - np.array(teaching_image_point) - np.array(shift_amount)

    res = calibration_with_2d_points(robot_points=relative_robot_positions,
                                     image_points=relative_image_positions)

    return res


# ===============================================================================
# NOTE: 描画系
# ===============================================================================
def draw_estimated_result(src: np.ndarray, pts: Union[list, np.ndarray], circle_: Optional[tuple] = None):
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

    points_ = np.round(pts).astype(np.int32)
    for pt in points_:
        dst = cv2.circle(
            img=dst,
            center=(pt[0], pt[1]),
            radius=thickness,
            color=color,
            thickness=-1
        )

    return dst


class HandEyeCalibration:
    @staticmethod
    def circle_estimation(pts: Optional[Union[list, np.ndarray]] = None,
                          x_data: Optional[Union[list, np.ndarray]] = None,
                          y_data: Optional[Union[list, np.ndarray]] = None) -> Optional[EstimatedCircle]:
        return circle_estimation(pts=pts, x_data=x_data, y_data=y_data)

    @staticmethod
    def calibration_with_2d_points(rbt_pts: Union[list, np.ndarray], img_pts: Union[list, np.ndarray]):
        return calibration_with_2d_points(robot_points=rbt_pts, image_points=img_pts)

    @staticmethod
    def draw_estimated_result(src: np.ndarray, pts: Union[list, np.ndarray], circle_: Optional[tuple] = None):
        return draw_estimated_result(src=src, pts=pts, circle_=circle_)


if __name__ == '__main__':
    ROBOT = [
        [200.130, 330.280],
        [453.980, 326.100],
        [200.130, 156.090],
        [453.980, 151.900],
    ]

    IM = [
        [1710.0, 1153.6],
        [339.6, 1108.7],
        [1725.7, 210.8],
        [354.4, 167.3]
    ]

    res = calibration_with_2d_points(ROBOT, IM)
    print(res)
