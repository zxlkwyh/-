
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import time
from collections import defaultdict

def check_dependencies():
    """检测核心库是否安装或损坏。"""
    missing = []
    corrupted = []
    
    # 检测 OpenCV 和 Numpy
    try:
        import numpy as np
        import cv2
    except ImportError as e:
        if "numpy" in str(e).lower():
            corrupted.append(f"Numpy (可能损坏: {e})")
        else:
            missing.append("opencv-python")
    except Exception as e:
        corrupted.append(f"环境异常: {e}")

    # 检测 Pillow
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing.append("Pillow")
        
    if missing or corrupted:
        error_msg = "检测到环境问题：\n"
        if missing: error_msg += f"- 缺失库: {', '.join(missing)}\n"
        if corrupted: error_msg += f"- 损坏库: {', '.join(corrupted)}\n"
        
        error_msg += f"\n请尝试运行以下修复命令:\n"
        error_msg += f"& {sys.executable} -m pip install --user --force-reinstall numpy==1.26.4 opencv-python Pillow"
        
        print(error_msg)
        if __name__ == "__main__":
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("环境异常", error_msg)
            sys.exit(1)
    return True

# 在加载核心库前进行依赖项检查
check_dependencies()

import cv2
from PIL import Image, ImageTk
from processing import build_log_path, process_frame, reset_state

# ==============================================================================
# 0. 全局样式配置
# ==============================================================================
UI_FONT = "楷体"

class RoundedButton(tk.Canvas):
    """
    自定义圆角按钮类。
    支持倒角、3D立体阴影效果、悬停高亮和点击反馈。
    """
    def __init__(self, parent, text, command=None, bg_color="#E3F2FD", fg_color="#0D47A1", 
                 width=120, height=40, radius=20, font=(UI_FONT, 10, "bold")):
        super().__init__(parent, width=width, height=height, bg=parent['bg'], 
                         highlightthickness=0, cursor="hand2")
        self.command = command
        self.text = text
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.radius = radius
        self.font = font
        self.width = width
        self.height = height
        
        # 绑定事件
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        self.draw_button(state="normal")

    def draw_button(self, state="normal"):
        self.delete("all")
        r = self.radius
        w = self.width
        h = self.height
        
        # 1. 绘制阴影 (立体感)
        shadow_offset = 3 if state != "pressed" else 1
        shadow_color = "#AAAAAA" if state != "pressed" else "#CCCCCC"
        self._draw_rounded_rect(shadow_offset, shadow_offset, w, h, r, shadow_color)
        
        # 2. 绘制主体
        body_offset = 0 if state != "pressed" else 2
        body_color = self.bg_color
        if state == "hover":
            # 悬停时颜色稍微加深或变亮
            body_color = self._adjust_color(self.bg_color, -10) 
            
        self._draw_rounded_rect(body_offset, body_offset, w-3+body_offset, h-3+body_offset, r, body_color)
        
        # 3. 绘制文字
        self.create_text(w/2 - 1.5 + body_offset, h/2 - 1.5 + body_offset, 
                         text=self.text, fill=self.fg_color, font=self.font)

    def _draw_rounded_rect(self, x1, y1, x2, y2, r, color):
        """绘制平滑的圆角矩形"""
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]
        return self.create_polygon(points, fill=color, smooth=True)

    def _adjust_color(self, hex_color, amount):
        """调整颜色的亮度"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = tuple(max(0, min(255, c + amount)) for c in rgb)
        return '#%02x%02x%02x' % new_rgb

    def _on_press(self, event):
        self.draw_button(state="pressed")
        if self.command:
            self.after(50, self.command) # 稍微延迟触发，让点击反馈更明显

    def _on_release(self, event):
        self.draw_button(state="hover")

    def _on_enter(self, event):
        self.draw_button(state="hover")

    def _on_leave(self, event):
        self.draw_button(state="normal")

# ==============================================================================
# 1. 系统初始化模块 - 全局常量字典 (特征表指定数值)
# ==============================================================================
# 这一组阈值会直接传给 processing.py，调参时优先改这里。
FEATURE_THRESHOLDS = {
    "day_brightness_mean": 145,
    "night_brightness_mean": 82,
    "outside_day_mean": 98,
    "outside_day_hysteresis": 90,
    "regular_light_total_min": 6,
    "regular_light_side_min": 4,
    "regular_light_band_score_min": 1.7,
    "regular_light_gap_cv_max": 0.95,
    "regular_light_y_min": 0.62,
    "regular_light_entry_ratio_min": 0.96,
    "regular_light_entry_ratio_max": 1.05,
    "regular_light_entry_saturation_override": 80,
    "regular_light_saturation_min": 48,
    "regular_light_saturation_max": 120,
    "ceiling_structure_stripe_min": 1,
    "ceiling_structure_tall_min": 1,
    "ceiling_structure_area_min": 650,
    "ceiling_structure_pair_min": 2,
    "ceiling_structure_area_strong_min": 2200,
    "ceiling_structure_height_min_ratio": 0.18,
    "ceiling_structure_aspect_min": 1.10,
    "tunnel_dark_hold_mean_max": 100,
    "tunnel_dark_hold_ratio_max": 0.90,
    "arch_area_ratio_min": 0.04,
    "arch_coverage_min": 0.18,
    "arch_depth_min": 0.05,
    "arch_center_offset_max": 0.22,
    "arch_width_ratio_min": 0.22,
    "arch_score_min": 1.35,
    "portal_reflector_segment_min": 4,
    "portal_reflector_coverage_min": 0.08,
    "arch_entry_roi_max": 0.90,
    "arch_entry_ratio_max": 1.00,
    "reentry_cooldown_frames": 10,
    "transition_display_frames": 20,
    "tunnel_entry_confirm_frames": 2,
    "arch_entry_confirm_frames": 8,
    "tunnel_support_release_frames": 6,
    "tunnel_day_exit_confirm_frames": 3,
    "tunnel_night_exit_confirm_frames": 2,
    "day_tunnel_min_frames": 120,
    "night_tunnel_min_frames": 180,
    "day_outside_min_gap_frames": 30,
    "night_outside_min_gap_frames": 35,
    "tunnel_exit_brightness_ratio_min": 1.03,
    "tunnel_exit_roi_min": 1.10,
    "hybrid_day_brightness_mean": 95,
    "hybrid_day_brightness_hysteresis": 88,
    "hybrid_entry_cooldown_frames": 40,
    "hybrid_light_entry_total_min": 10,
    "hybrid_light_entry_side_min": 6,
    "hybrid_light_entry_alignment_min": 2.2,
    "hybrid_light_entry_line_min": 12,
    "hybrid_light_entry_y_min": 0.69,
    "hybrid_light_entry_saturation_max": 85,
    "hybrid_light_entry_ratio_min": 0.98,
    "hybrid_light_entry_roi_min": 0.95,
    "hybrid_dark_entry_roi_min": 1.05,
    "hybrid_dark_entry_change_max": -0.03,
    "hybrid_dark_entry_mean_max": 130,
    "hybrid_dark_entry_y_min": 0.72,
    "hybrid_dark_entry_light_total_min": 2,
    "hybrid_dark_entry_line_min": 20,
    "hybrid_portal_regular_total_min": 6,
    "hybrid_portal_regular_side_min": 4,
    "hybrid_portal_regular_alignment_min": 2.1,
    "hybrid_portal_regular_line_min": 18,
    "hybrid_portal_regular_y_min": 0.66,
    "hybrid_portal_regular_roi_min": 0.90,
    "hybrid_portal_regular_ratio_min": 0.95,
    "hybrid_portal_regular_saturation_min": 50,
    "hybrid_portal_regular_saturation_max": 90,
    "hybrid_entry_delay_frames": 5,
    "hybrid_entry_fast_delay_frames": 1,
    "hybrid_entry_fast_roi_min": 1.10,
    "hybrid_inside_light_total_min": 10,
    "hybrid_inside_light_side_min": 6,
    "hybrid_inside_alignment_min": 2.2,
    "hybrid_inside_line_min": 14,
    "hybrid_inside_y_min": 0.69,
    "hybrid_inside_mean_min": 45,
    "hybrid_direct_inside_mean_max": 133,
    "hybrid_direct_inside_roi_min": 0.88,
    "hybrid_inside_saturation_max": 90,
    "hybrid_hold_light_total_min": 5,
    "hybrid_hold_side_min": 4,
    "hybrid_hold_alignment_min": 1.6,
    "hybrid_hold_line_min": 10,
    "hybrid_hold_y_min": 0.70,
    "hybrid_hold_mean_max": 115,
    "hybrid_hold_roi_min": 1.00,
    "hybrid_hold_saturation_max": 90,
    "hybrid_dark_hold_mean_max": 105,
    "hybrid_dark_hold_ratio_max": 0.85,
    "hybrid_dark_hold_roi_min": 1.10,
    "hybrid_day_uniform_center_texture_max": 22,
    "hybrid_day_uniform_side_max": 5,
    "hybrid_day_uniform_alignment_max": 2.0,
    "hybrid_day_uniform_y_max": 0.67,
    "hybrid_day_uniform_exit_confirm_frames": 1,
    "hybrid_exit_ratio_min": 1.15,
    "hybrid_exit_mean_min": 110,
    "hybrid_exit_alignment_max": 1.9,
    "hybrid_exit_confirm_frames": 4,
    "entry_confirm_frames": 2,
    "transition_entry_confirm_frames": 2,
    "portal_entry_confirm_frames": 1,
    "portal_entry_cooldown_frames": 20,
    "entry_roi_contrast_max": 0.90,
    "entry_brightness_ratio_max": 0.86,
    "entry_sky_edge_max": 0.045,
    "entry_saturation_max": 92,
    "entry_texture_max": 38,
    "entry_center_texture_max": 82,
    "entrance_display_delay_frames": 5,
    "entrance_display_roi_contrast_max": 0.88,
    "entrance_display_brightness_ratio_max": 0.94,
    "portal_dark_roi_min": 1.05,
    "portal_dark_brightness_ratio_max": 0.99,
    "portal_dark_texture_max": 26,
    "portal_dark_center_texture_max": 55,
    "portal_dark_line_count_min": 20,
    "portal_dark_sky_edge_max": 0.015,
    "portal_bright_roi_min": 1.17,
    "portal_bright_brightness_ratio_min": 1.06,
    "portal_bright_texture_max": 12,
    "portal_bright_center_texture_max": 18,
    "portal_bright_line_count_min": 20,
    "portal_bright_sky_edge_max": 0.01,
    "inside_brightness_max": 162,
    "inside_roi_contrast_max": 0.92,
    "inside_saturation_max": 95,
    "inside_sky_edge_max": 0.05,
    "inside_line_count_min": 2,
    "inside_texture_max": 40,
    "inside_center_texture_max": 90,
    "texture_hold_max": 35,
    "texture_hold_center_texture_max": 95,
    "texture_hold_brightness_max": 145,
    "texture_hold_saturation_max": 75,
    "texture_hold_brightness_ratio_max": 1.05,
    "deep_tunnel_texture_max": 14,
    "deep_tunnel_center_texture_max": 20,
    "deep_tunnel_brightness_ratio_max": 1.20,
    "deep_tunnel_line_count_min": 4,
    "deep_tunnel_light_count_min": 8,
    "tunnel_light_total_min": 7,
    "tunnel_light_side_min": 3,
    "tunnel_light_alignment_min": 1.8,
    "tunnel_light_y_mean_min": 0.62,
    "tunnel_light_saturation_max": 95,
    "tunnel_light_brightness_ratio_max": 1.02,
    "tunnel_light_texture_max": 40,
    "tunnel_light_center_texture_max": 95,
    "exit_roi_contrast_min": 1.20,
    "exit_brightness_ratio_min": 1.29,
    "exit_sky_edge_min": 0.05,
    "exit_saturation_min": 80,
    "exit_confirm_frames": 2,
    "event_hold_frames": 3,
}

# 默认把选视频窗口指到桌面的“视频”文件夹，省一次手动定位。
VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "视频"))

class App(tk.Tk):
    """主应用程序类，负责UI构建、事件处理和整体调度。"""
    def __init__(self):
        """初始化应用程序窗口和所有状态变量。"""
        super().__init__()
        self.title("软件设计")
        self.geometry("1200x800")
        self.configure(bg="white")
        
        # 设置全局默认字体为楷体
        self.option_add("*Font", (UI_FONT, 10))
        self.option_add("*Background", "white")
        
        # --- 视频和摄像头状态变量 ---
        self.cap = None                 # OpenCV视频捕获对象
        self.is_playing = False         # 视频/摄像头是否正在播放
        self.is_dragging = False        # 用户是否正在拖动进度条
        self.video_path = None          # 当前加载的视频文件路径
        self.total_frames = 0           # 视频总帧数
        self.current_frame_num = 0      # 当前处理的帧号
        self.fps = 0                    # 视频的帧率

        # --- 统计数据变量 ---
        self.scene_counts = defaultdict(int) # 统计各种场景出现的帧数
        self.processing_times = []           # 记录每帧的处理耗时
        self.analysis_log_path = build_log_path("video_session")

        self.create_widgets()

    def create_widgets(self):
        """创建并布局所有UI组件。"""
        # --- 窗口网格布局配置 ---
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ==============================================================================
        # 9.1 窗口布局（5大区域）
        # ==============================================================================

        # --- 1. 顶部：模式与开发依据提示区 ---
        top_frame = tk.Frame(self, bg="white", height=60, relief=tk.GROOVE, borderwidth=2)
        top_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        tk.Label(top_frame, text="视频信号", bg="white", font=(UI_FONT, 11, "bold")).pack(side=tk.LEFT, padx=20)
        
        self.mode_var = tk.StringVar(value="video") # 模式选择变量，默认为视频模式
        
        video_mode_btn = tk.Radiobutton(top_frame, text="视频测试模式", variable=self.mode_var, value="video", command=self.switch_mode, bg="white", selectcolor="white")
        video_mode_btn.pack(side=tk.RIGHT, padx=20)
        
        cam_mode_btn = tk.Radiobutton(top_frame, text="摄像头实时采集模式", variable=self.mode_var, value="camera", command=self.switch_mode, bg="white", selectcolor="white")
        cam_mode_btn.pack(side=tk.RIGHT, padx=20)

        # --- 2. 中部左侧：原始视频帧 ---
        self.mid_left_frame = tk.LabelFrame(self, text=" 原始帧 ", bg="white", relief="groove", borderwidth=2, font=(UI_FONT, 10, "bold"))
        self.mid_left_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.raw_frame_label = tk.Label(self.mid_left_frame, bg="black")
        self.raw_frame_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # --- 3. 中部右侧：预处理分析视图 ---
        self.mid_right_frame = tk.LabelFrame(self, text=" 预处理分析 ", bg="white", relief="groove", borderwidth=2, font=(UI_FONT, 10, "bold"))
        self.mid_right_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        self.current_view_type = tk.StringVar(value="morph") # 默认显示形态学特征

        # 预处理视图主体标签 (大尺寸)
        self.processed_frame_label = tk.Label(self.mid_right_frame, bg="black")
        self.processed_frame_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # --- 右侧：核心信息展示区 ---
        right_info_frame = tk.Frame(self, width=280, bg="white", relief="groove", borderwidth=2)
        right_info_frame.grid(row=1, column=2, sticky="ns", padx=10, pady=10)
        right_info_frame.grid_propagate(False)

        tk.Label(right_info_frame, text="核心信息展示区", font=(UI_FONT, 16, "bold"), bg="white", fg="#333333").pack(pady=15, anchor="w", padx=10)

        self.scene_result_var = tk.StringVar(value="场景识别结果: --")
        tk.Label(right_info_frame, textvariable=self.scene_result_var, font=(UI_FONT, 14, "bold"), fg="#003366", bg="white").pack(pady=8, anchor="w", padx=10)

        self.light_status_var = tk.StringVar(value="近光灯: -- | 远光灯: --")
        tk.Label(right_info_frame, textvariable=self.light_status_var, font=(UI_FONT, 12), bg="white", fg="#444444").pack(pady=8, anchor="w", padx=10)

        self.validation_result_var = tk.StringVar(value="光照适配: --")
        self.validation_label = tk.Label(right_info_frame, textvariable=self.validation_result_var, font=(UI_FONT, 12, "bold"), bg="white")
        self.validation_label.pack(pady=8, anchor="w", padx=10)

        # --- 视频测试专属UI ---
        self.video_controls_frame = tk.Frame(right_info_frame, bg="white")
        self.video_controls_frame.pack(pady=20, fill='x', anchor="w", padx=10)
        
        tk.Label(self.video_controls_frame, text="视频进度控制:", font=(UI_FONT, 10, "bold"), bg="white").pack(anchor="w")
        
        # 使用自定义样式的进度条
        style = ttk.Style()
        style.configure("TScale", background="white")
        
        self.progress_bar = ttk.Scale(self.video_controls_frame, from_=0, to=100, orient="horizontal", command=self._on_progress_drag, style="TScale")
        self.progress_bar.pack(fill="x", pady=10)
        self.progress_bar.bind("<ButtonPress-1>", self._on_progress_press)
        self.progress_bar.bind("<ButtonRelease-1>", self._on_progress_release)

        self.frame_label_var = tk.StringVar(value="帧: 0/0 | 剩余: 0")
        tk.Label(self.video_controls_frame, textvariable=self.frame_label_var, bg="white", font=(UI_FONT, 9)).pack(anchor="w")

        # --- 4. 底部：功能操作区 ---
        bottom_frame = tk.Frame(self, bg="white", height=120, relief=tk.GROOVE, borderwidth=2)
        bottom_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # 按钮容器 (左侧)
        left_btn_container = tk.Frame(bottom_frame, bg="white")
        left_btn_container.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 1. 模式特定按钮组 (视频/摄像头)
        self.mode_btns_container = tk.Frame(left_btn_container, bg="white")
        self.mode_btns_container.pack(side=tk.LEFT)

        self.cam_buttons_frame = tk.Frame(self.mode_btns_container, bg="white")
        RoundedButton(self.cam_buttons_frame, " 开始采集 ", self.start_camera, bg_color="#E1F5FE", fg_color="#01579B").pack(side=tk.LEFT, padx=5)
        RoundedButton(self.cam_buttons_frame, " 暂停采集 ", self.pause_camera, bg_color="#FFF9C4", fg_color="#F57F17").pack(side=tk.LEFT, padx=5)

        self.video_buttons_frame = tk.Frame(self.mode_btns_container, bg="white")
        RoundedButton(self.video_buttons_frame, " 选择视频 ", self.select_video, bg_color="#E3F2FD", fg_color="#0D47A1").pack(side=tk.LEFT, padx=5)
        RoundedButton(self.video_buttons_frame, " 播放/暂停 ", self.play_pause_video, bg_color="#E8F5E9", fg_color="#1B5E20").pack(side=tk.LEFT, padx=5)
        RoundedButton(self.video_buttons_frame, " 重新上传 ", self.select_video, bg_color="#F5F5F5", fg_color="#424242").pack(side=tk.LEFT, padx=5)

        # 2. 分割线 (可选，增加视觉区分)
        tk.Frame(left_btn_container, width=2, bg="#CCCCCC").pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=10)

        # 3. 通用视图切换按钮组
        self.view_btns_frame = tk.Frame(left_btn_container, bg="white")
        self.view_btns_frame.pack(side=tk.LEFT)
        
        RoundedButton(self.view_btns_frame, " 灰度视图 ", lambda: self.current_view_type.set("gray"), bg_color="#F5F5F5", fg_color="#333333").pack(side=tk.LEFT, padx=5)
        RoundedButton(self.view_btns_frame, " 边缘检测 ", lambda: self.current_view_type.set("canny"), bg_color="#F5F5F5", fg_color="#333333").pack(side=tk.LEFT, padx=5)
        RoundedButton(self.view_btns_frame, " 开运算 ", lambda: self.current_view_type.set("opening"), bg_color="#F5F5F5", fg_color="#333333").pack(side=tk.LEFT, padx=5)
        RoundedButton(self.view_btns_frame, " 特征叠加 ", lambda: self.current_view_type.set("morph"), bg_color="#E3F2FD", fg_color="#0D47A1").pack(side=tk.LEFT, padx=5)

        # 退出按钮 (中间靠右)
        RoundedButton(bottom_frame, " 退出系统 ", self.quit_app, bg_color="#FFEBEE", fg_color="#B71C1C").pack(side=tk.RIGHT, padx=20, pady=20)

        # --- 5. 右下角：核心阈值提示区 (使用 pack 避免遮挡) ---
        bottom_right_frame = tk.LabelFrame(bottom_frame, text=" 软件设计 ", bg="#FFFFF0", relief=tk.RIDGE, borderwidth=2, font=(UI_FONT, 10, "bold"))
        bottom_right_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        tk.Label(bottom_right_frame, text="视频信号识别隧道", font=(UI_FONT, 9), bg="#FFFFF0").pack(anchor="w", padx=10, pady=5)
        
        self.switch_mode() # 根据默认模式初始化UI

    def _show_error(self, error_type, message, solution):
        """显示格式统一的错误弹窗。"""
        messagebox.showerror(error_type, f"错误信息: {message}\n\n解决方案: {solution}")

    def _reset_runtime_state(self, source_name):
        """重置一次视频/摄像头会话，避免上一段识别状态串入新任务。"""
        self.scene_counts = defaultdict(int)
        self.processing_times = []
        self.current_frame_num = 0

        # 每次会话单独生成一个 TXT，后面查问题更清楚。
        self.analysis_log_path = build_log_path(source_name)
        reset_state(self.analysis_log_path)

        self.scene_result_var.set("场景识别结果: --")
        self.light_status_var.set("近光灯: -- | 远光灯: --")
        self.validation_result_var.set("光照适配: --")
        self.validation_label.config(fg="#444444")
        self.update_frame_label()

    # ==============================================================================
    # 8. 视频进度拖动处理模块
    # ==============================================================================
    def _on_progress_press(self, event):
        """当用户点击或触摸进度条时，暂停视频播放。"""
        if self.cap is None: return
        self.is_dragging = True
        self.pause_video()

    def _on_progress_drag(self, value):
        """在拖动过程中，实时更新帧号标签。"""
        if self.is_dragging and self.cap is not None:
            target_frame = int(float(value))
            self.current_frame_num = target_frame
            self.update_frame_label()

    def _on_progress_release(self, event):
        """当用户释放进度条时，跳转到目标帧。"""
        if self.cap is None: return
        self.is_dragging = False
        target_frame = int(self.progress_bar.get())
        self.jump_to_frame(target_frame)

    def jump_to_frame(self, frame_num):
        """跳转到视频的指定帧并执行全流程处理。"""
        if self.cap is None:
            self._show_error("进度拖动异常", "视频未加载", "请先选择一个视频文件")
            return
        
        original_frame = self.current_frame_num
        try:
            # 限制帧号在有效范围内
            frame_num = max(0, min(frame_num, self.total_frames - 1))
            self.current_frame_num = frame_num
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("无法读取目标帧")

            # 处理并显示目标帧
            results = self._process_and_display_frame(frame, is_jump=True)
            # 按格式要求在控制台打印拖动记录
            print(f"【视频模式 - 进度拖动】从帧 {original_frame} 跳转到帧 {self.current_frame_num} | 目标帧识别场景（按特征表）：{results['scene']} | 灯光模拟：近光灯 {results['light_status']['low_beam']} 远光灯 {results['light_status']['high_beam']} | 反馈验证：{results['validation']}")
        except Exception as e:
            self.current_frame_num = original_frame # 跳转失败时恢复原帧号
            self._show_error("进度拖动异常", f"跳转到帧 {frame_num} 失败: {e}", f"请尝试在 0-{self.total_frames-1} 范围内拖动")

    # ==============================================================================
    # 4.2 本地视频上传测试模式
    # ==============================================================================
    def select_video(self):
        """打开文件对话框让用户选择视频，并初始化视频播放器。"""
        self.pause_video() # 暂停当前播放
        if self.cap:
            self.cap.release()
            self.cap = None

        # 默认从项目外层的视频目录开始选，和老师给的测试素材位置一致。
        path = filedialog.askopenfilename(
            initialdir=VIDEO_DIR if os.path.isdir(VIDEO_DIR) else os.path.dirname(__file__),
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if not path: # 如果用户取消选择
            return

        try:
            self.video_path = path
            # 重新选视频时，必须先把状态机和日志文件都切成新的会话。
            self._reset_runtime_state(os.path.splitext(os.path.basename(path))[0])
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise IOError(f"无法打开视频文件: {self.video_path}")

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames == 0:
                raise ValueError("视频总帧数为0或文件已损坏")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0: self.fps = 30 # 如果获取不到帧率，则默认为30
            
            self.progress_bar.config(to=self.total_frames - 1)
            self.progress_bar.set(0)
            
            # 读取并显示第一帧
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("无法读取视频的第一帧")

            # 第一帧先显示出来，用户不用等播放才看到画面。
            self._process_and_display_frame(frame)

        except (IOError, ValueError, Exception) as e:
            self._show_error("视频上传异常", str(e), "请选择一个有效的、未损坏的视频文件 (mp4/avi/mov)")
            if self.cap:
                self.cap.release()
            self.cap = None # 失败时确保cap对象被清空

    def play_pause_video(self):
        """切换视频的播放和暂停状态。"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        """开始或恢复视频播放。"""
        if self.cap is None:
            self._show_error("播放错误", "没有加载视频", "请先选择一个视频文件")
            return
        self.is_playing = True
        self.update_video() # 启动视频更新循环

    def pause_video(self):
        """暂停视频播放。"""
        self.is_playing = False

    def update_video(self):
        """视频播放的主循环，负责逐帧读取、处理和调度显示。"""
        if not self.is_playing or self.cap is None or self.is_dragging:
            return
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame_num += 1

                # 到末尾就停下来，并把摘要信息打印到控制台。
                # 检查是否到达视频末尾
                if self.current_frame_num >= self.total_frames:
                    self.is_playing = False
                    self.current_frame_num = self.total_frames - 1
                    self.update_frame_label()
                    self.progress_bar.set(self.current_frame_num)
                    self._show_summary() # 播放结束时显示摘要
                    return

                self._process_and_display_frame(frame)
                self.progress_bar.set(self.current_frame_num)
                # 根据视频帧率安排下一次更新
                self.after(int(1000 / self.fps), self.update_video)
            else:
                # 视频流结束
                self.is_playing = False
                self._show_summary()
        except Exception as e:
            self.is_playing = False
            self._show_error("视频流异常", f"在播放过程中读取帧失败: {e}", "视频文件可能已损坏或被移动")

    # ==============================================================================
    # 10. 主函数调度模块
    # ==============================================================================
    def _process_and_display_frame(self, frame, is_jump=False):
        """调度核心处理流程并更新UI，是连接后台与前端的桥梁。"""
        try:
            start_time = time.time()
            results = process_frame(frame, FEATURE_THRESHOLDS) # 调用外部处理模块
            end_time = time.time()
            
            # 如果不是拖动跳转，则记录统计数据
            if not is_jump:
                self.processing_times.append((end_time - start_time) * 1000) # 毫秒
                self.scene_counts[results['scene']] += 1

            # 这里把后端处理结果统一映射回右侧文字区和两块图像区。
            self._show_frame_on_ui(frame, results['processed_views'])
            self.scene_result_var.set(f"场景识别结果: {results['scene']}")
            low_beam = results['light_status']['low_beam']
            high_beam = results['light_status']['high_beam']
            self.light_status_var.set(f"近光灯: {low_beam} | 远光灯: {high_beam}")
            validation = results['validation']
            self.validation_result_var.set(f"光照适配: {validation}")
            self.validation_label.config(fg="green" if validation == "达标" else "red")
            self.update_frame_label()
            
            # 控制台保留一份简洁的逐帧记录，调试时很好用。
            if not is_jump:
                mode_name = "摄像头模式" if self.mode_var.get() == "camera" else "视频模式"
                total_text = "实时流" if self.total_frames == 0 else f"{self.total_frames-1}"
                print(
                    f"【{mode_name}】帧号：{self.current_frame_num} / {total_text} | "
                    f"原始场景：{results['raw_scene']} | 稳定场景：{results['scene']} | "
                    f"弧形锁定：{results['features']['is_locked']} | 近光灯：{low_beam} | "
                    f"亮度：{results['features']['brightness_mean']:.1f} | "
                    f"纹理度：{results['features']['texture_score']:.1f} | "
                    f"中心纹理：{results['features']['center_texture_score']:.1f}"
                )
            
            return results
        except Exception as e:
            self.pause_video()
            self._show_error("处理异常", f"处理帧 {self.current_frame_num} 时发生错误: {e}", "请检查处理逻辑或视频帧数据")
            return None # 指示处理失败

    def _show_summary(self):
        """在视频播放结束后，向控制台输出摘要统计信息。"""
        print("\n" + "="*50)
        print("【视频测试完成】")
        if self.total_frames > 0:
            print(f"共解析 {self.total_frames} 帧")
            # 打印每种场景的统计
            for scene, count in self.scene_counts.items():
                print(f"| {scene}：{count} 帧")
            
            # 计算并打印平均耗时
            avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            print(f"| 平均识别反馈耗时：{avg_time:.2f}ms")
            print(f"| 帧级参数日志：{self.analysis_log_path}")
        print("="*50 + "\n")

    # ==============================================================================
    # 9. 图形UI可视化模块 (辅助函数)
    # ==============================================================================
    def _show_frame_on_ui(self, raw_frame, processed_views):
        """将原始帧和用户选择的预处理视图显示在UI上。"""
        # 1. 显示原始帧 (大图)
        self._display_image(self.raw_frame_label, raw_frame, (640, 480))
        
        # 2. 根据底部按钮选择，切换灰度 / 边缘 / 开运算 / 特征叠加视图
        view_type = self.current_view_type.get()
        selected_view = processed_views.get(view_type, processed_views['morph'])
        
        self._display_image(self.processed_frame_label, selected_view, (640, 480))

    def _display_image(self, label, frame, target_size):
        """将OpenCV图像转换为Tkinter图像并显示在指定的Label上。"""
        display_w, display_h = target_size
        frame_h, frame_w = frame.shape[:2]

        # 计算缩放比例以保持纵横比
        scale = min(display_w / frame_w, display_h / frame_h)
        dim = (int(frame_w * scale), int(frame_h * scale))
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # 创建一个黑色画布，并将缩放后的图像粘贴到中心
        canvas = Image.new("RGB", (display_w, display_h), "black")
        
        # 转换为 PIL 图像 (OpenCV 是 BGR, PIL 是 RGB)
        img_pil = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_pil)
        
        paste_x = (display_w - dim[0]) // 2
        paste_y = (display_h - dim[1]) // 2
        canvas.paste(img_pil, (paste_x, paste_y))

        # 转换为Tkinter PhotoImage并更新Label
        photo_img = ImageTk.PhotoImage(image=canvas)
        label.config(image=photo_img)
        label.image = photo_img # 保持对图像的引用

    def update_frame_label(self):
        """更新显示当前帧/总帧数和剩余帧数的标签。"""
        if self.total_frames > 0:
            remaining_frames = self.total_frames - self.current_frame_num
            self.frame_label_var.set(f"帧: {self.current_frame_num}/{self.total_frames-1} | 剩余: {remaining_frames-1}")
        elif self.mode_var.get() == "camera" and self.current_frame_num > 0:
            self.frame_label_var.set(f"实时帧: {self.current_frame_num}")
        else:
            self.frame_label_var.set("帧: 0/0 | 剩余: 0")

    # ==============================================================================
    # 4.1 摄像头实时采集模式
    # ==============================================================================
    def start_camera(self):
        """启动摄像头并开始实时采集和处理。"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None

            # 摄像头模式也单独建一份日志，方便和视频测试分开看。
            self._reset_runtime_state("camera_live")
            self.cap = cv2.VideoCapture(0) # 0 代表默认摄像头
            if not self.cap.isOpened():
                raise IOError("无法连接到摄像头")
            self.total_frames = 0 # 摄像头模式下总帧数为0
            self.is_playing = True
            self.update_camera_feed() # 启动摄像头更新循环
        except (IOError, Exception) as e:
            self._show_error("硬件异常", str(e), "请检查摄像头连接并确保驱动已正确安装")
            self.cap = None

    def update_camera_feed(self):
        """摄像头采集的主循环。"""
        if not self.is_playing or self.mode_var.get() != 'camera':
            return
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame_num += 1
            self._process_and_display_frame(frame)
            self.after(20, self.update_camera_feed) # 大约 50fps 的刷新率
        else:
            self.is_playing = False
            self._show_error("视频流中断", "无法从摄像头读取帧", "请检查摄像头是否被其他应用占用")

    def pause_camera(self):
        """暂停摄像头采集。"""
        self.is_playing = False

    # ==============================================================================
    # 辅助功能函数
    # ==============================================================================
    def switch_mode(self):
        """切换工作模式（视频测试 vs 摄像头采集）。"""
        self.pause_video()
        self.pause_camera()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 切模式时把画面、统计和状态机一起清空，避免旧结果残留在界面上。
        self.total_frames = 0
        self.current_frame_num = 0
        self.scene_counts = defaultdict(int)
        self.processing_times = []
        reset_state(build_log_path(f"{self.mode_var.get()}_idle"))
        self.scene_result_var.set("场景识别结果: --")
        self.light_status_var.set("近光灯: -- | 远光灯: --")
        self.validation_result_var.set("光照适配: --")
        self.validation_label.config(fg="#444444")
        self.update_frame_label()
        self.progress_bar.set(0)
        self.raw_frame_label.config(image='')
        self.processed_frame_label.config(image='')

        mode = self.mode_var.get()
        # 更新单选按钮的视觉样式
        for child in self.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Radiobutton):
                        if btn['value'] == mode:
                            btn.config(fg="#0D47A1", font=(UI_FONT, 10, "bold"))
                        else:
                            btn.config(fg="#666666", font=(UI_FONT, 10, "normal"))

        # 根据模式显示/隐藏对应的按钮和控件
        if mode == "camera":
            self.cam_buttons_frame.pack(side=tk.LEFT, padx=10)
            self.video_buttons_frame.pack_forget()
            self.video_controls_frame.pack_forget()
        else: # video mode
            self.video_buttons_frame.pack(side=tk.LEFT, padx=10)
            self.cam_buttons_frame.pack_forget()
            self.video_controls_frame.pack(pady=20, fill='x', anchor="w", padx=10)

    def quit_app(self):
        """安全退出应用程序，释放资源。"""
        if self.cap:
            self.cap.release()
        self.quit()
        self.destroy()

if __name__ == "__main__":
    try:
        # 针对 Windows 系统，通过调用 Windows API 提升高 DPI 屏幕下的清晰度
        # 这可以解决字体和UI元素在高分屏上模糊或有“马赛克感”的问题
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)

        app = App()
        app.mainloop()
    except Exception as e:
        # 捕获任何未处理的顶层异常
        messagebox.showerror("UI异常", f"应用程序启动失败: {e}")
        print(f"[UI异常] 应用程序启动失败: {e}")
