# -*- coding: utf-8 -*-
"""
MARS_UART_capture.py
====================
從 IWR6843 雷達透過 UART 即時擷取點雲資料。

工作流程：
  1. 開啟兩個 Serial ports（CONFIG_PORT、DATA_PORT）
  2. 送出 config 檔案設定雷達
  3. 讀取 TLV frame，解析點雲
  4. 即時視覺化 + 儲存為 .mat 檔

用法：
  # 預設設定（COM7/COM8，200 frames）
  python MARS_UART_capture.py

  # 自訂參數
  python MARS_UART_capture.py --cfg xwr68xx_MARS_UART_test.cfg \\
                               --port_cfg COM7 --port_data COM8 \\
                               --frames 200 --output mars_pointcloud_0506.mat

設定檔：
  xwr68xx_MARS_UART_test.cfg （需置於同目錄）

輸出檔案：
  mars_pointcloud_0506.mat
  資料維度：(N_FRAMES, 64, 5)
  5 channels = [x, y, z, doppler, intensity]

硬體：
  IWR6843 MARS 雷達
  CONFIG_PORT  : 115200 baud（設定用）
  DATA_PORT    : 921600 baud（點雲用）
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import serial
import struct
from scipy.io import savemat

import util.get_abs_dir as get_abs_dir
path_project_root, path_feature, path_pointcloud = get_abs_dir.get_abs_dir()

# ============================================================
# 常數定義
# ============================================================
TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO       = 7

# Frame header 同步碼（TI mmWave 協議固定值）
FRAME_SYNC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# Frame header 結構
FRAME_HEADER_SIZE = 48  # bytes
FRAME_LENGTH_OFFSET = 12  # bytes（相對於 sync word)

# TLV header 結構
TLV_TYPE_SIZE = 2  # bytes
TLV_LENGTH_SIZE = 2  # bytes
TLV_HEADER_SIZE = TLV_TYPE_SIZE + TLV_LENGTH_SIZE
POINT_DATA_SIZE = 20  # bytes (x,y,z,doppler,intensity 各 4 bytes float)

# ============================================================
# 預設設定
# ============================================================
path_cfg_dir = os.path.join(path_project_root, 'get_pointcloud', 'cfg')
DEFAULT_CFG_FILE   = 'iwr6843_clutter_removal.cfg'
DEFAULT_CFG_FILE   = os.path.normpath(os.path.join(path_cfg_dir, DEFAULT_CFG_FILE))
DEFAULT_PORT_CFG   = 'COM7'
DEFAULT_PORT_DATA  = 'COM8'
DEFAULT_FRAMES     = 200
DEFAULT_OUTPUT     = 'mars_pointcloud_0.mat'


# ============================================================
# 核心函數
# ============================================================
def find_frame_header(buffer, sync_word):
    """在 buffer 中尋找 frame header 的同步碼位置"""
    for i in range(len(buffer) - len(sync_word) + 1):
        if buffer[i:i+len(sync_word)] == sync_word:
            return i
    return -1


def parse_tlv_frame(frame_bytes, tlv_type_points, tlv_type_info):
    """
    解析 TLV frame，提取點雲資料
    
    Args:
        frame_bytes: 完整的 frame bytes
        tlv_type_points: 點雲 TLV type ID
        tlv_type_info: 側資訊 TLV type ID
    
    Returns:
        ndarray: (N, 5) 其中 5 = [x, y, z, doppler, intensity]
    """
    if len(frame_bytes) < FRAME_HEADER_SIZE:
        return None
    
    # 跳過 frame header，開始解析 TLV
    tlv_offset = FRAME_HEADER_SIZE
    points = []
    
    while tlv_offset < len(frame_bytes) - TLV_HEADER_SIZE:
        tlv_type = struct.unpack('<H', frame_bytes[tlv_offset:tlv_offset+2])[0]
        tlv_len  = struct.unpack('<H', frame_bytes[tlv_offset+2:tlv_offset+4])[0]
        tlv_offset += TLV_HEADER_SIZE
        
        if tlv_type == tlv_type_points:
            # 點雲資料：(x, y, z, doppler, intensity) 各 4 bytes float
            n_points = tlv_len // POINT_DATA_SIZE
            for i in range(n_points):
                offset = tlv_offset + i * POINT_DATA_SIZE
                x, y, z, dop, inten = struct.unpack(
                    '<fffff',
                    frame_bytes[offset:offset+POINT_DATA_SIZE]
                )
                points.append([x, y, z, dop, inten])
        
        tlv_offset += tlv_len
    
    return np.array(points, dtype=np.float32) if points else None


def to_mars_format(raw_points, max_points=64):
    """
    將原始點雲轉成 MARS 格式 (max_points, 5)
    不足補零，超過就截斷
    """
    mars_frame = np.zeros((max_points, 5), dtype=np.float32)
    
    if raw_points is not None and len(raw_points) > 0:
        n = min(len(raw_points), max_points)
        mars_frame[:n] = raw_points[:n]
    
    return mars_frame


# ============================================================
# UART 讀取類別
# ============================================================
class RadarUARTReader:
    def __init__(self, cfg_port, data_port, baudrate_cfg=115200, baudrate_data=921600):
        self.cfg_port = serial.Serial(cfg_port, baudrate=baudrate_cfg, timeout=0.1)
        self.data_port = serial.Serial(data_port, baudrate=baudrate_data, timeout=0.1)
        self.byte_buffer = b''
        self.frame_sync_word = FRAME_SYNC_WORD
    
    def send_config(self, cfg_file):
        """送出 config 檔案至雷達"""
        print('[INFO] 送出 config...')
        with open(cfg_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                self.cfg_port.write((line + '\n').encode())
                print(f'  [CFG] {line}')
                import time; time.sleep(0.05)
        print('[INFO] Config 送出完成。\n')
    
    def read_frame(self):
        """
        讀取一個完整 frame，回傳點雲
        
        Returns:
            ndarray: (N, 5) 點雲資料
        """
        while True:
            # 讀取新 bytes
            n_avail = self.data_port.in_waiting
            if n_avail > 0:
                self.byte_buffer += self.data_port.read(n_avail)
            
            # 防止 buffer 無限增長
            if len(self.byte_buffer) > 65536:
                self.byte_buffer = self.byte_buffer[-32768:]
            
            # 找 frame 同步碼
            frame_start_idx = find_frame_header(self.byte_buffer, self.frame_sync_word)
            if frame_start_idx < 0:
                import time; time.sleep(0.005)
                continue
            
            # 確認 frame length 欄位已到達
            if len(self.byte_buffer) < frame_start_idx + FRAME_LENGTH_OFFSET + 4:
                import time; time.sleep(0.005)
                continue
            
            # 讀取 frame 長度
            frame_len = struct.unpack(
                '<I',
                self.byte_buffer[frame_start_idx+FRAME_LENGTH_OFFSET:frame_start_idx+FRAME_LENGTH_OFFSET+4]
            )[0]
            
            # 確認完整 frame 已到達
            if len(self.byte_buffer) < frame_start_idx + frame_len:
                import time; time.sleep(0.005)
                continue
            
            # 擷取並移除已讀的 frame
            frame_bytes = self.byte_buffer[frame_start_idx:frame_start_idx+frame_len]
            self.byte_buffer = self.byte_buffer[frame_start_idx+frame_len:]
            
            # 解析點雲
            raw_points = parse_tlv_frame(
                frame_bytes,
                TLV_DETECTED_POINTS,
                TLV_SIDE_INFO
            )
            
            if raw_points is not None:
                return raw_points
    
    def close(self):
        """關閉 serial ports"""
        self.cfg_port.close()
        self.data_port.close()


# ============================================================
# 主程式
# ============================================================
def get_next_output_path(output_path):
    """若檔案已存在，依序產生 mars_pointcloud_1.mat、mars_pointcloud_2.mat ..."""
    base, ext = os.path.splitext(output_path)
    idx = 0

    if not os.path.exists(output_path):
        return output_path

    while True:
        idx += 1
        new_path = f"{base}_{idx}{ext}"
        if not os.path.exists(new_path):
            return new_path


def main():
    parser = argparse.ArgumentParser(description='IWR6843 MARS UART 點雲擷取')
    parser.add_argument('--cfg', default=DEFAULT_CFG_FILE,
                        help='Config 檔案')
    parser.add_argument('--port_cfg', default=DEFAULT_PORT_CFG,
                        help='Config serial port')
    parser.add_argument('--port_data', default=DEFAULT_PORT_DATA,
                        help='Data serial port')
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES,
                        help='要擷取的 frame 數')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='輸出 .mat 檔案')
    args = parser.parse_args()

    # 檢查 config 檔案是否存在
    if not os.path.isfile(args.cfg):
        print(f'[ERROR] Config 檔案不存在: {args.cfg}')
        sys.exit(1)
    
    # 從 config 檔名提取名稱（不含副檔名）
    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    
    # 若輸出檔案用預設值，改成 mars_pointcloud_0_{cfg_name}.mat
    if args.output == DEFAULT_OUTPUT:
        base, ext = os.path.splitext(DEFAULT_OUTPUT)
        args.output = f'{base}_{cfg_name}{ext}'
    
    args.output = get_next_output_path(args.output)
    print(f'[INFO] Config 檔案: {args.cfg}')
    print(f'[INFO] 輸出檔案: {args.output}\n')

    # ============================================================
    # 初始化
    # ============================================================
    print('[INFO] 初始化 UART...')
    reader = RadarUARTReader(args.port_cfg, args.port_data)
    
    print('[INFO] 送出 config...')
    reader.send_config(args.cfg)
    
    mars_data = np.zeros((args.frames, 64, 5), dtype=np.float32)
    
    # ============================================================
    # 即時視覺化設定
    # ============================================================
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle('MARS UART 即時點雲擷取', fontsize=12, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter([], [], [], c=[], cmap='turbo', s=36, vmin=0, vmax=50)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 3)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=15, azim=-60)
    
    plt.colorbar(scatter, ax=ax, label='Intensity')
    
    # ============================================================
    # 主迴圈
    # ============================================================
    print(f'[INFO] 開始錄製，目標 {args.frames} frames。')
    print('[INFO] 請站在雷達前方 1~2 公尺處。\n')
    
    frame_count = 0
    
    try:
        while frame_count < args.frames:
            raw_points = reader.read_frame()
            mars_frame = to_mars_format(raw_points)
            
            frame_count += 1
            mars_data[frame_count-1] = mars_frame
            
            # 統計
            valid_idx = np.any(mars_frame != 0, axis=1)
            n_pts = np.sum(valid_idx)
            
            print(f'Frame {frame_count:3d}/{args.frames} | 有效點數: {n_pts:2d}')
            
            # 更新視覺化
            if plt.fignum_exists(fig.number):
                scatter._offsets3d = (mars_frame[:, 0], mars_frame[:, 1], mars_frame[:, 2])
                scatter.set_array(mars_frame[:, 4])
                ax.set_title(f'Frame {frame_count}/{args.frames} | 有效點數：{n_pts}')
                plt.pause(0.01)
    
    except KeyboardInterrupt:
        print('\n[INFO] 使用者中斷。')
    
    finally:
        reader.close()
        plt.close(fig)
    
    # ============================================================
    # 儲存
    # ============================================================
    mars_data = mars_data[:frame_count]
    savemat(args.output, {'marsData': mars_data})
    
    print(f'\n[INFO] 錄製完成，共 {frame_count} frames。')
    print(f'[INFO] 已儲存至 {args.output}')
    print(f'[INFO] 資料維度：[{frame_count} × 64 × 5]')
    
    valid_counts = np.sum(np.any(mars_data != 0, axis=2), axis=1)
    print(f'[INFO] 平均有效點數：{np.mean(valid_counts):.1f}')
    print(f'[INFO] 最大有效點數：{np.max(valid_counts)}')
    print(f'[INFO] 最小有效點數：{np.min(valid_counts)}')
    
    print(f'\n[下一步] python pc_to_featuremap_v2_mars.py --input {args.output}')


if __name__ == '__main__':
    main()