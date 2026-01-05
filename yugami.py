import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors

class AdvancedStarTrailCalibrator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.vis_image = cv2.imread(image_path) # デバッグ表示用
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        self.h, self.w = self.image.shape[:2]
        self.points = None

    def preprocess_and_extract_points(self):
        """
        地上物や文字を除去し、純粋な星の軌跡のみを抽出する強化された前処理
        """
        print("Preprocessing: Masking ground/text and extracting trails...")
        
        # 1. マスク処理: 画像の下部15%（地上や日付）を強制的に無視する
        mask_height = int(self.h * 0.85)
        roi_image = self.image.copy()
        roi_image[mask_height:, :] = 0  # 下部を黒塗り
        
        # 2. 二値化
        _, binary = cv2.threshold(roi_image, 40, 255, cv2.THRESH_BINARY)

        # 3. ノイズ除去: 小さな点（ホットピクセルや短いノイズ）を削除
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        min_area = 20 # 面積が小さいゴミを除去
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = 0

        # 4. モルフォロジー変換（点線をつなぐ）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=2)
        
        # 5. 細線化
        skeleton = skeletonize(dilated > 0)
        y_coords, x_coords = np.where(skeleton)
        
        # 点が少なすぎる場合はエラー
        if len(x_coords) < 100:
            raise ValueError("Not enough star trail points found. Threshold might be too high.")

        # ランダムサンプリング（計算量削減のため、最大3000点に絞る）
        if len(x_coords) > 3000:
            idx = np.random.choice(len(x_coords), 3000, replace=False)
            x_coords = x_coords[idx]
            y_coords = y_coords[idx]

        self.points = np.column_stack((x_coords, y_coords)).astype(np.float32)
        
        # デバッグ用: 抽出された点を赤色で描画して保存
        for p in self.points:
            cv2.circle(self.vis_image, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
        
        # 近傍点モデル構築
        self.nbrs_model = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(self.points)
        print(f"Extracted {len(self.points)} points for optimization.")

    def _undistort_points_fisheye(self, points, K, D):
        pts_reshaped = points.reshape(-1, 1, 2)
        # fisheye.undistortPointsは正規化座標(x/z, y/z)を返すため、fを掛ける必要はないが
        # 半径計算のために焦点距離でスケーリングしたほうが安定する場合がある
        undistorted = cv2.fisheye.undistortPoints(pts_reshaped, K, D)
        return undistorted.reshape(-1, 2)

    def _objective_function(self, params):
        f, cx, cy = params[0], params[1], params[2]
        k1, k2, k3, k4 = params[3:7]
        pole_x, pole_y = params[7], params[8]

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([k1, k2, k3, k4], dtype=np.float32)

        try:
            # 点群と天の極をアンディストート
            points_u = self._undistort_points_fisheye(self.points, K, D)
            pole_input = np.array([[pole_x, pole_y]], dtype=np.float32).reshape(-1, 1, 2)
            pole_u = cv2.fisheye.undistortPoints(pole_input, K, D).reshape(2)
        except:
            return np.ones(len(self.points) * 5) * 1e6

        # 半径を計算
        radii = np.linalg.norm(points_u - pole_u, axis=1)

        # 近傍点との半径差を最小化
        distances, indices = self.nbrs_model.kneighbors(self.points)
        residuals = []
        
        # ロバスト性を高めるため、Huber Loss的な考え方で極端な外れ値の影響を抑える処理を入れることも可能だが
        # ここでは単純差分を行う
        for i in range(len(self.points)):
            neighbor_indices = indices[i, 1:]
            # 半径の差
            radius_diffs = radii[i] - radii[neighbor_indices]
            residuals.extend(radius_diffs)

        return np.array(residuals)

    def calibrate(self):
        if self.points is None:
            self.preprocess_and_extract_points()

        print("Optimizing lens parameters...")

        # --- 初期値の改善 ---
        # 画像を見ると天の極は「右上」にある。(x ~ 1400, y ~ 400) あたりと推測
        initial_pole_x = self.w * 0.75
        initial_pole_y = self.h * 0.35
        
        initial_f = self.w * 0.6  # 魚眼なので焦点距離は短めからスタート
        initial_c = [self.w / 2.0, self.h / 2.0]
        initial_k = [0.0, 0.0, 0.0, 0.0]
        
        x0 = np.array([initial_f, *initial_c, *initial_k, initial_pole_x, initial_pole_y])

        # パラメータ範囲制約
        # poleの位置も画像の範囲外少しまで許容
        lower_bound = [100, 0, 0, -5, -5, -5, -5, -self.w, -self.h]
        upper_bound = [self.w*2, self.w, self.h, 5, 5, 5, 5, self.w*2, self.h*2]

        res = least_squares(self._objective_function, x0, bounds=(lower_bound, upper_bound), 
                            method='trf', verbose=1, ftol=1e-4, loss='soft_l1') # soft_l1で外れ値に強くする

        optimized_params = res.x
        
        f, cx, cy = optimized_params[0], optimized_params[1], optimized_params[2]
        self.K_opt = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        self.D_opt = np.array(optimized_params[3:7], dtype=np.float32)
        
        print(f"Opt finish. Cost: {res.cost:.2f}")
        print(f"Params: f={f:.1f}, c=({cx:.1f},{cy:.1f}), D={self.D_opt}")
        print(f"Pole: ({optimized_params[7]:.1f}, {optimized_params[8]:.1f})")

        # デバッグ用：推定された極の位置を描画
        cv2.circle(self.vis_image, (int(optimized_params[7]), int(optimized_params[8])), 10, (255, 0, 0), -1)

    def save_results(self, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. デバッグ画像の保存
            cv2.imwrite(os.path.join(output_dir, "debug_points.jpg"), self.vis_image)

            print("Generating optimized distortion maps...")

            # --- 改善点: 出力画像の最適化 ---
            # OpenCVの機能を使って、画像全体が入る最適なカメラ行列(new_K)を計算させます
            # balance=0.0 : 黒枠を削除してズームイン（情報は減るが見た目は良い）
            # balance=1.0 : 全画角を含める（黒枠が出る）
            # ここではバランスを0.5くらいにして、適度な範囲を狙います
            balance = 0.5 
            
            # 魚眼用の新しいカメラ行列を推定
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K_opt, self.D_opt, (self.w, self.h), np.eye(3), balance=balance
            )
            
            # マップ生成
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K_opt, self.D_opt, np.eye(3), new_K, (self.w, self.h), cv2.CV_16SC2
            )
            
            # 保存
            np.save(os.path.join(output_dir, 'distortion_map_x.npy'), map1)
            np.save(os.path.join(output_dir, 'distortion_map_y.npy'), map2)

            # プレビュー画像作成
            undistorted_img = cv2.remap(self.image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # 出力パス
            preview_path = os.path.join(output_dir, "preview_undistorted_fixed.jpg")
            cv2.imwrite(preview_path, undistorted_img)
            
            print(f"Results saved to {output_dir}")
            print(f"Check '{preview_path}' for the improved view.")

if __name__ == '__main__':
    input_path = r"C:\Users\kekke\Downloads\test.jpg"
    output_dir = os.path.join(os.path.dirname(input_path), "star_trail_v2_output")

    calib = AdvancedStarTrailCalibrator(input_path)
    calib.calibrate()
    calib.save_results(output_dir)