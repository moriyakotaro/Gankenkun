#!/usr/bin/env python3
"""
プレビュー制御による二足歩行ロボットの重心軌道生成
線形倒立振子モデル(LIPM)を使用
"""
import math
import numpy as np
import control
import control.matlab
import csv
from typing import List, Tuple, Optional

class preview_control:
    """プレビュー制御クラス (既存コードとの互換性を維持)"""
    
    def __init__(self, dt: float, period: float, z: float, Q: float = 1.0e+8, H: float = 1.0):
        """
        初期化
        
        Parameters:
        -----------
        dt : float
            サンプリング時間 [s]
        period : float
            プレビュー期間 [s]
        z : float
            重心高さ [m]
        Q : float
            状態コスト重み
        H : float
            入力コスト重み
        """
        self.dt = dt
        self.period = period
        self.z = z
        G = 9.8  # 重力加速度
        
        # 連続時間システムの定義 (x, dx, ddx)
        A = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ])
        B = np.array([[0.0], [0.0], [1.0]])
        C = np.array([[1.0, 0.0, -z/G]])  # ZMP方程式: p = x - z/g * ddx
        D = 0
        
        # 離散化
        sys = control.matlab.ss(A, B, C, D)
        sys_d = control.c2d(sys, dt)
        self.A_d, self.B_d, self.C_d, D_d = control.matlab.ssdata(sys_d)
        
        # 拡大系の構築
        E_d = np.array([[dt], [1.0], [0.0]])
        Zero = np.zeros((3, 1))
        
        # 拡大系の状態方程式: [e, x]^T (eは出力誤差の積分)
        Phai = np.block([
            [1.0, -self.C_d @ self.A_d],
            [Zero, self.A_d]
        ])
        G_mat = np.block([
            [-self.C_d @ self.B_d],
            [self.B_d]
        ])
        GR = np.block([[1.0], [Zero]])
        Gd = np.block([
            [-self.C_d @ E_d],
            [E_d]
        ])
        
        # コスト行列
        Qm = np.zeros((4, 4))
        Qm[0, 0] = Q
        
        # 離散代数リカッチ方程式を解く
        P = control.dare(Phai, G_mat, Qm, H)[0]
        
        # 最適フィードバックゲイン
        self.F = -np.linalg.inv(H + G_mat.T @ P @ G_mat) @ G_mat.T @ P @ Phai
        
        # プレビューゲインの計算
        xi = (np.eye(4) - G_mat @ np.linalg.inv(H + G_mat.T @ P @ G_mat) @ G_mat.T @ P) @ Phai
        
        self.f = []
        preview_steps = round(period / dt)
        for i in range(preview_steps):
            f_i = -np.linalg.inv(H + G_mat.T @ P @ G_mat) @ G_mat.T @ np.linalg.matrix_power(xi.T, i) @ P @ GR
            self.f.append(f_i)
        
        # 状態変数の初期化
        self.xp = np.zeros((3, 1))
        self.yp = np.zeros((3, 1))
        self.ux = 0.0
        self.uy = 0.0
    
    def set_param(self, t: float, current_x, current_y, 
                  foot_plan: List[List[float]], pre_reset: bool = False):
        """
        重心軌道を計算
        
        Parameters:
        -----------
        t : float
            現在時刻 [s]
        current_x : np.ndarray or np.matrix
            x方向の現在状態 [位置, 速度, 加速度]
        current_y : np.ndarray or np.matrix
            y方向の現在状態 [位置, 速度, 加速度]
        foot_plan : List[List[float]]
            足配置計画 [[時刻, x座標, y座標], ...]
        pre_reset : bool
            状態をリセットするかどうか
        
        Returns:
        --------
        COG_X : List of np.ndarray
            重心軌道 [x_cog, y_cog, x_zmp, y_zmp] の配列
        x : np.ndarray
            更新後のx方向状態
        y : np.ndarray
            更新後のy方向状態
        """
        # 入力がmatrixの場合はarrayに変換
        if isinstance(current_x, np.matrix):
            x = np.asarray(current_x)
        else:
            x = current_x.copy()
        
        if isinstance(current_y, np.matrix):
            y = np.asarray(current_y)
        else:
            y = current_y.copy()
        
        if pre_reset:
            self.xp = x.copy()
            self.yp = y.copy()
            self.ux = 0.0
            self.uy = 0.0
        
        COG_X = []
        steps = round((foot_plan[1][0] - t) / self.dt)
        
        for i in range(steps):
            # 現在のZMP位置
            px = (self.C_d @ x)[0, 0]
            py = (self.C_d @ y)[0, 0]
            
            # ZMP誤差
            ex = foot_plan[0][1] - px
            ey = foot_plan[0][2] - py
            
            # 拡大系の状態ベクトル
            X = np.block([[ex], [x - self.xp]])
            Y = np.block([[ey], [y - self.yp]])
            
            self.xp = x.copy()
            self.yp = y.copy()
            
            # フィードバック制御入力
            dux = (self.F @ X)[0, 0]
            duy = (self.F @ Y)[0, 0]
            
            # プレビュー制御入力
            index = 1
            for j in range(len(self.f)):
                current_step = round((i + j + t / self.dt))
                if index < len(foot_plan) and current_step >= round(foot_plan[index][0] / self.dt):
                    dux += (self.f[j] * (foot_plan[index][1] - foot_plan[index-1][1]))[0, 0]
                    duy += (self.f[j] * (foot_plan[index][2] - foot_plan[index-1][2]))[0, 0]
                    index += 1
            
            self.ux += dux
            self.uy += duy
            
            # 状態更新
            x = self.A_d @ x + self.B_d * self.ux
            y = self.A_d @ y + self.B_d * self.uy
            
            # 元のコードとの互換性のため2次元配列で返す
            COG_X.append(np.array([[x[0, 0], y[0, 0], px, py]]))
        
        return COG_X, x, y


def main():
    """メイン実行関数"""
    # パラメータ設定
    dt = 0.01  # 10ms
    preview_period = 1.0  # 1秒先まで予見
    cog_height = 0.27  # 重心高さ 27cm
    
    pc = preview_control(dt, preview_period, cog_height)
    
    # 足配置計画 [時刻, x座標, y座標]
    foot_step = [
        [0.00, 0.00, 0.00],
        [0.34, 0.00, 0.06],
        [0.68, 0.05, -0.04],
        [1.02, 0.10, 0.10],
        [1.36, 0.15, 0.00],
        [1.70, 0.20, 0.14],
        [2.04, 0.25, 0.10],
        [2.72, 0.25, 0.10],
        [100.0, 0.25, 0.10]  # 終端
    ]
    
    # 初期状態
    x = np.zeros((3, 1))
    y = np.zeros((3, 1))
    
    # 結果をCSVに保存
    with open('result.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_cog', 'y_cog', 'x_zmp', 'y_zmp'])
    
    t = 0
    step_count = 0
    
    while len(foot_step) > 2:
        print(f"Step {step_count}: t={t:.2f}s, Remaining steps: {len(foot_step)-1}")
        
        # 重心軌道計算
        cog, x, y = pc.set_param(t, x, y, foot_step)
        
        # CSV書き込み
        with open('result.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(cog)
        
        # 次のステップへ
        del foot_step[0]
        t = foot_step[0][0]
        step_count += 1
    
    print(f"\n完了! 結果は result.csv に保存されました")
    print(f"総ステップ数: {step_count}")


if __name__ == '__main__':
    main()
