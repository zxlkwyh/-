import os
import re
from collections import deque

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_FILE = os.path.join(BASE_DIR, "tunnel_frame_metrics.txt")

# 这些阈值集中放在这里，后面调参时只需要改这一处。
DEFAULT_THRESHOLDS = {
    "night_brightness_mean": 82,
    "day_brightness_mean": 145,
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


class SceneState:
    def __init__(self):
        # 最近几帧的平滑判决缓存，避免场景一闪一闪地跳。
        self.decision_history = []
        self.history_size = 6

        # 亮度历史主要用于判断白天 / 黑夜，以及做相对亮度对比。
        self.brightness_history = []
        self.long_term_brightness = 150.0
        self.environment = "白天"

        self.dark_area_history = deque(maxlen=10)
        self.bright_area_history = deque(maxlen=10)
        self.dark_growth_ema = 0.0
        self.bright_growth_ema = 0.0
        self.ema_alpha = 0.42

        self.in_tunnel_session = False
        self.last_exit_time = -999
        self.entrance_detected_frames = 0
        self.entrance_pending_display = False
        self.entrance_display_counter = 0
        self.pre_entry_scene = "白天"
        self.exit_pending_display = False
        self.exit_display_counter = 0
        self.post_exit_scene = "白天"
        self.exit_detected_frames = 0
        self.non_tunnel_frames = 0
        self.tunnel_session_frames = 0
        self.outside_session_frames = 0

        self.consecutive_arc_frames = 0
        self.arc_lost_frames = 0
        self.last_target_pos = None
        self.roi_coords = None
        self.frame_count = 0

        self.current_stable_scene = "白天"
        self.current_base_scene = "白天"
        self.event_hold_frames = 0
        self.stable_transition_frames = 0
        self.pending_scene = None
        self.confidence = 50


state = SceneState()
LOG_FILE = DEFAULT_LOG_FILE


def build_log_path(source_name="session"):
    # 视频名里可能带空格或特殊字符，这里先做一次清洗，免得写日志时报路径问题。
    safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", source_name).strip("_")
    if not safe_name:
        safe_name = "session"
    return os.path.join(BASE_DIR, f"{safe_name}_frame_metrics.txt")


def reset_state(log_file=None):
    global state, LOG_FILE
    # 每次切换视频或切换模式时，都要把状态机彻底清干净。
    state = SceneState()
    LOG_FILE = log_file or DEFAULT_LOG_FILE


def _merge_thresholds(thresholds):
    merged = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        merged.update(thresholds)
    return merged


def _calc_light_band_metrics(points, roi_height):
    metrics = {
        "count": len(points),
        "alignment_score": 0.0,
        "alignment_residual": 999.0,
        "gap_cv": 999.0,
        "band_score": 0.0,
    }
    if len(points) < 3:
        return metrics

    pts = np.array(points, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    if float(np.std(ys)) < 6.0:
        return metrics

    coef = np.polyfit(ys, xs, 1)
    predicted = coef[0] * ys + coef[1]
    residual = float(np.std(xs - predicted))
    alignment_score = len(points) / (1.0 + residual / 12.0)

    ordered_y = np.sort(ys)
    min_gap = max(4.0, roi_height * 0.025)
    max_gap = roi_height * 0.35
    gaps = np.diff(ordered_y)
    gaps = gaps[(gaps >= min_gap) & (gaps <= max_gap)]

    if len(gaps) >= 2:
        gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1.0))
    elif len(gaps) == 1:
        gap_cv = 0.0
    else:
        gap_cv = 999.0

    gap_penalty = 1.0 + min(gap_cv, 2.0) if gap_cv < 900 else 3.0
    band_score = alignment_score / gap_penalty

    metrics.update(
        {
            "alignment_score": alignment_score,
            "alignment_residual": residual,
            "gap_cv": gap_cv,
            "band_score": band_score,
        }
    )
    return metrics


def _detect_portal_arch(gray_frame, mean_val):
    h, w = gray_frame.shape
    x1, x2 = int(w * 0.14), int(w * 0.86)
    y1, y2 = int(h * 0.18), int(h * 0.78)
    roi = gray_frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape
    roi_area = max(1, roi_h * roi_w)

    dark_thresh = int(max(35, min(110, mean_val * 0.78)))
    _, portal_mask = cv2.threshold(roi, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    portal_mask = cv2.morphologyEx(
        portal_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    portal_mask = cv2.morphologyEx(
        portal_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    )

    contours, _ = cv2.findContours(portal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = {
        "score": 0.0,
        "area_ratio": 0.0,
        "coverage": 0.0,
        "depth": 0.0,
        "center_offset": 1.0,
        "width_ratio": 0.0,
        "residual": 999.0,
        "bbox": None,
        "mask": np.zeros_like(gray_frame),
    }

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < roi_area * 0.02:
            continue

        x, y, ww, hh = cv2.boundingRect(contour)
        width_ratio = float(ww / max(roi_w, 1))
        height_ratio = float(hh / max(roi_h, 1))
        center_x = (x + ww / 2) / max(roi_w, 1)
        center_offset = abs(center_x - 0.5)
        if width_ratio < 0.18 or height_ratio < 0.18 or center_offset > 0.30:
            continue

        contour_mask = np.zeros_like(portal_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        valid_columns = np.where(np.any(contour_mask > 0, axis=0))[0]
        if len(valid_columns) < roi_w * 0.16:
            continue

        top_boundary = []
        for col in valid_columns:
            ys = np.where(contour_mask[:, col] > 0)[0]
            if ys.size > 0:
                top_boundary.append(float(ys[0]))

        if len(top_boundary) < 8:
            continue

        top_boundary = np.array(top_boundary, dtype=np.float32)
        x_norm = (valid_columns.astype(np.float32) / max(roi_w - 1, 1)) - 0.5
        y_norm = top_boundary / max(roi_h - 1, 1)

        if len(valid_columns) >= 12:
            coef = np.polyfit(x_norm, y_norm, 2)
            fitted = coef[0] * x_norm * x_norm + coef[1] * x_norm + coef[2]
            residual = float(np.std(y_norm - fitted))
        else:
            residual = 0.25

        left_part = y_norm[x_norm < -0.18]
        center_part = y_norm[np.abs(x_norm) <= 0.10]
        right_part = y_norm[x_norm > 0.18]
        if left_part.size == 0 or center_part.size == 0 or right_part.size == 0:
            continue

        arch_depth = float(((np.mean(left_part) + np.mean(right_part)) * 0.5) - np.mean(center_part))
        coverage = float(len(valid_columns) / max(roi_w, 1))
        area_ratio = float(area / roi_area)
        score = (
            area_ratio * 3.0
            + coverage * 2.0
            + width_ratio
            + arch_depth * 6.0
            - center_offset * 2.0
            - residual * 6.0
        )

        if score <= best["score"]:
            continue

        full_mask = np.zeros_like(gray_frame)
        full_mask[y1:y2, x1:x2] = contour_mask
        best = {
            "score": score,
            "area_ratio": area_ratio,
            "coverage": coverage,
            "depth": arch_depth,
            "center_offset": center_offset,
            "width_ratio": width_ratio,
            "residual": residual,
            "bbox": (x1 + x, y1 + y, ww, hh),
            "mask": full_mask,
        }

    return best


def _measure_portal_reflectors(gray_frame, portal_mask_full, bbox, mean_val):
    metrics = {"segment_count": 0, "coverage": 0.0}
    if bbox is None:
        return metrics

    x, y, w, h = bbox
    if w <= 12 or h <= 12:
        return metrics

    gray_roi = gray_frame[y:y + h, x:x + w]
    mask_roi = portal_mask_full[y:y + h, x:x + w]
    if gray_roi.size == 0 or mask_roi.size == 0:
        return metrics

    ring = cv2.dilate(mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    ring = cv2.subtract(ring, mask_roi)
    upper_limit = max(1, int(h * 0.42))
    ring[upper_limit:, :] = 0
    if np.count_nonzero(ring) == 0:
        return metrics

    local_upper = gray_roi[:upper_limit, :]
    local_thr = int(np.percentile(local_upper, 78)) if local_upper.size > 0 else int(mean_val)
    bright_thr = max(int(mean_val * 1.08), local_thr)
    _, bright_mask = cv2.threshold(gray_roi, bright_thr, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.bitwise_and(bright_mask, ring)
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segment_count = 0
    total_width = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (6 <= area <= 600):
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if by + bh > upper_limit or bw < 3 or bh < 2:
            continue
        segment_count += 1
        total_width += bw

    metrics["segment_count"] = segment_count
    metrics["coverage"] = float(total_width / max(w, 1))
    return metrics


def preprocess_frame(frame):
    h, w = frame.shape[:2]
    new_w = 640
    new_h = max(1, int(h * (new_w / w)))
    resized = cv2.resize(frame, (new_w, new_h))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 上半部分一般是天空，单独压一下噪声，避免白天云层把边缘图搅乱。
    sky_ratio = 0.38
    sky_limit = int(new_h * sky_ratio)
    if sky_limit > 10:
        gray[0:sky_limit, :] = cv2.GaussianBlur(gray[0:sky_limit, :], (5, 5), 2.0)

    ground_region = gray[sky_limit:, :]
    if ground_region.size > 0:
        # 地面区域稍微增强一点局部对比度，更利于提线条和暗区。
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray[sky_limit:, :] = clahe.apply(ground_region)

    return resized, gray


def extract_features(color_frame, gray_frame, thresholds):
    state.frame_count += 1
    h, w = gray_frame.shape
    total_pixels = h * w

    # 亮度、纹理、饱和度是后面场景判别最常用的基础特征。
    mean_val = float(np.mean(gray_frame))
    std_val = float(np.std(gray_frame))
    texture_score = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
    # 全图纹理有时会被地面或车灯带偏，所以再额外量一下中部区域纹理，专门区分隧道内外。
    upper_texture_roi = gray_frame[int(h * 0.12):int(h * 0.46), int(w * 0.18):int(w * 0.82)]
    center_texture_roi = gray_frame[int(h * 0.28):int(h * 0.74), int(w * 0.20):int(w * 0.80)]
    upper_texture_score = float(cv2.Laplacian(upper_texture_roi, cv2.CV_64F).var()) if upper_texture_roi.size > 0 else texture_score
    center_texture_score = float(cv2.Laplacian(center_texture_roi, cv2.CV_64F).var()) if center_texture_roi.size > 0 else texture_score
    hsv_small = cv2.cvtColor(color_frame[::4, ::4], cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv_small[:, :, 1]))

    prev_mean = state.brightness_history[-1] if state.brightness_history else mean_val
    state.brightness_history.append(mean_val)
    if len(state.brightness_history) > 20:
        state.brightness_history.pop(0)

    recent_avg = np.mean(state.brightness_history[-10:]) if len(state.brightness_history) >= 10 else mean_val
    state.environment = "黑夜" if recent_avg < thresholds["night_brightness_mean"] else "白天"

    # 长期亮度基线不要跟着瞬时波动乱跑，在隧道内外用不同速度更新更稳。
    if not state.in_tunnel_session:
        state.long_term_brightness = state.long_term_brightness * 0.97 + mean_val * 0.03
    else:
        state.long_term_brightness = state.long_term_brightness * 0.99 + mean_val * 0.01

    # Canny 阈值用图像统计量自适应，兼顾白天和黑夜。
    sigma = 0.33
    median_val = np.median(gray_frame)
    canny_low = int(max(0, (1.0 - sigma) * median_val))
    canny_high = int(min(255, (1.0 + sigma) * median_val))
    canny_high = max(canny_low + 20, canny_high)
    canny_high = min(255, canny_high)

    if state.environment == "白天":
        canny_low = max(30, canny_low)
        canny_high = max(canny_low + 30, canny_high)
    else:
        canny_low = max(15, min(40, canny_low))
        canny_high = max(canny_low + 15, min(100, canny_high))

    edges = cv2.Canny(gray_frame, canny_low, canny_high)
    edge_density = float(np.sum(edges > 0) / total_pixels)

    sky_height = int(h * 0.28)
    sky_pixel_count = max(1, sky_height * w)
    sky_edge_density = float(np.sum(edges[0:sky_height, :] > 0) / sky_pixel_count)
    edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 上方小灯点直接按“单侧成带 + 间距相对稳定”来判断，不再混入太多别的特征。
    light_roi_top = int(h * 0.08)
    light_roi_bottom = int(h * 0.58)
    light_roi = gray_frame[light_roi_top:light_roi_bottom, :]
    light_tophat = cv2.morphologyEx(
        light_roi,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    )
    light_thr = max(20, int(np.percentile(light_tophat, 99.15)))
    _, light_mask = cv2.threshold(light_tophat, light_thr, 255, cv2.THRESH_BINARY)
    light_mask = cv2.morphologyEx(
        light_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    light_contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_light_mask = np.zeros_like(light_mask)
    light_points = []
    left_light_points = []
    right_light_points = []

    for contour in light_contours:
        area = cv2.contourArea(contour)
        if not (4 <= area <= 180):
            continue

        x, y, ww, hh = cv2.boundingRect(contour)
        cx = x + ww / 2
        cy = y + hh / 2

        if cy > light_mask.shape[0] * 0.82:
            continue
        if cx < w * 0.08 or cx > w * 0.92:
            continue

        cv2.drawContours(filtered_light_mask, [contour], -1, 255, -1)
        light_points.append((cx, cy))
        if cx < w * 0.42:
            left_light_points.append((cx, cy))
        elif cx > w * 0.58:
            right_light_points.append((cx, cy))

    left_metrics = _calc_light_band_metrics(left_light_points, light_mask.shape[0])
    right_metrics = _calc_light_band_metrics(right_light_points, light_mask.shape[0])
    dominant_points = left_light_points if left_metrics["band_score"] >= right_metrics["band_score"] else right_light_points
    dominant_metrics = left_metrics if left_metrics["band_score"] >= right_metrics["band_score"] else right_metrics

    tunnel_light_total_count = len(light_points)
    tunnel_light_side_count = dominant_metrics["count"]
    tunnel_light_alignment_score = dominant_metrics["alignment_score"]
    tunnel_light_alignment_residual = dominant_metrics["alignment_residual"]
    tunnel_light_gap_cv = dominant_metrics["gap_cv"]
    tunnel_light_band_score = dominant_metrics["band_score"]
    tunnel_light_y_mean = (
        float(np.mean([point[1] / max(1, light_mask.shape[0]) for point in dominant_points]))
        if dominant_points
        else 0.0
    )

    tunnel_light_presence = (
        tunnel_light_total_count >= thresholds["regular_light_total_min"]
        and tunnel_light_side_count >= thresholds["regular_light_side_min"]
        and tunnel_light_band_score >= thresholds["regular_light_band_score_min"]
        and tunnel_light_gap_cv <= thresholds["regular_light_gap_cv_max"]
        and tunnel_light_y_mean >= thresholds["regular_light_y_min"]
        and thresholds["regular_light_saturation_min"] <= saturation <= thresholds["regular_light_saturation_max"]
    )

    light_mask_full = np.zeros_like(gray_frame)
    light_mask_full[light_roi_top:light_roi_bottom, :] = filtered_light_mask

    # 隧道内的顶部边缘往往会出现连续的纵向/弯曲长条结构，洞外通常没有这类密集肋条。
    structure_roi_top = int(h * 0.14)
    structure_roi_bottom = int(h * 0.58)
    structure_roi = edges[structure_roi_top:structure_roi_bottom, :]
    structure_side_mask = np.zeros_like(structure_roi)
    left_limit = int(w * 0.38)
    right_start = int(w * 0.62)
    structure_side_mask[:, :left_limit] = structure_roi[:, :left_limit]
    structure_side_mask[:, right_start:] = structure_roi[:, right_start:]
    structure_mask = cv2.morphologyEx(
        structure_side_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
        iterations=1,
    )
    structure_mask = cv2.dilate(
        structure_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
        iterations=1,
    )

    structure_contours, _ = cv2.findContours(structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ceiling_structure_mask_roi = np.zeros_like(structure_mask)
    ceiling_structure_stripe_count = 0
    ceiling_structure_tall_count = 0
    ceiling_structure_left_count = 0
    ceiling_structure_right_count = 0
    ceiling_structure_area = 0.0
    ceiling_structure_max_height = 0
    min_structure_height = int(structure_mask.shape[0] * thresholds["ceiling_structure_height_min_ratio"])

    for contour in structure_contours:
        area = cv2.contourArea(contour)
        x, y, ww, hh = cv2.boundingRect(contour)
        aspect = hh / max(ww, 1)
        if (
            hh < min_structure_height
            or area < 20
            or aspect < thresholds["ceiling_structure_aspect_min"]
        ):
            continue

        ceiling_structure_stripe_count += 1
        ceiling_structure_area += area
        ceiling_structure_max_height = max(ceiling_structure_max_height, hh)
        center_x = x + ww / 2
        if center_x < w * 0.38:
            ceiling_structure_left_count += 1
        elif center_x > w * 0.62:
            ceiling_structure_right_count += 1
        if hh >= int(structure_mask.shape[0] * 0.28):
            ceiling_structure_tall_count += 1
        cv2.drawContours(ceiling_structure_mask_roi, [contour], -1, 255, -1)

    ceiling_structure_presence = (
        ceiling_structure_stripe_count >= thresholds["ceiling_structure_stripe_min"]
        and (
            ceiling_structure_stripe_count >= thresholds["ceiling_structure_pair_min"]
            or (
                ceiling_structure_tall_count >= thresholds["ceiling_structure_tall_min"]
                and ceiling_structure_area >= thresholds["ceiling_structure_area_strong_min"]
            )
        )
    )

    ceiling_structure_mask_full = np.zeros_like(gray_frame)
    ceiling_structure_mask_full[structure_roi_top:structure_roi_bottom, :] = ceiling_structure_mask_roi

    # 洞口拱形只看“前方中心的大块暗区 + 上沿呈拱形”。
    portal_metrics = _detect_portal_arch(gray_frame, mean_val)
    portal_arch_score = portal_metrics["score"]
    portal_area_ratio = portal_metrics["area_ratio"]
    portal_coverage = portal_metrics["coverage"]
    portal_depth = portal_metrics["depth"]
    portal_center_offset = portal_metrics["center_offset"]
    portal_width_ratio = portal_metrics["width_ratio"]
    portal_mask_full = portal_metrics["mask"]
    reflector_metrics = _measure_portal_reflectors(gray_frame, portal_mask_full, portal_metrics["bbox"], mean_val)
    portal_reflector_segment_count = reflector_metrics["segment_count"]
    portal_reflector_coverage = reflector_metrics["coverage"]

    arch_presence = (
        portal_arch_score >= thresholds["arch_score_min"]
        and portal_area_ratio >= thresholds["arch_area_ratio_min"]
        and portal_coverage >= thresholds["arch_coverage_min"]
        and portal_depth >= thresholds["arch_depth_min"]
        and portal_center_offset <= thresholds["arch_center_offset_max"]
        and portal_width_ratio >= thresholds["arch_width_ratio_min"]
    )

    if arch_presence:
        state.consecutive_arc_frames = min(state.consecutive_arc_frames + 1, 10)
        state.arc_lost_frames = 0
    else:
        state.arc_lost_frames += 1
        if state.arc_lost_frames >= 2:
            state.consecutive_arc_frames = 0

    dark_metric = float(np.count_nonzero(portal_mask_full))
    bright_metric = float(np.count_nonzero(filtered_light_mask))
    state.dark_area_history.append(dark_metric)
    state.bright_area_history.append(bright_metric)

    if len(state.dark_area_history) >= 2:
        growth = (state.dark_area_history[-1] - state.dark_area_history[-2]) / (state.dark_area_history[-2] + 1e-6) * 100
        state.dark_growth_ema = state.ema_alpha * growth + (1 - state.ema_alpha) * state.dark_growth_ema
    if len(state.bright_area_history) >= 2:
        growth = (state.bright_area_history[-1] - state.bright_area_history[-2]) / (state.bright_area_history[-2] + 1e-6) * 100
        state.bright_growth_ema = state.ema_alpha * growth + (1 - state.ema_alpha) * state.bright_growth_ema

    if portal_metrics["bbox"] is not None:
        rx, ry, rw, rh = portal_metrics["bbox"]
        roi_gray = gray_frame[ry:ry + rh, rx:rx + rw]
        roi_mean = float(np.mean(roi_gray)) if roi_gray.size > 0 else mean_val
    else:
        rx, ry = int(w * 0.25), int(h * 0.28)
        rw, rh = int(w * 0.50), int(h * 0.36)
        roi_gray = gray_frame[ry:ry + rh, rx:rx + rw]
        roi_mean = float(np.mean(roi_gray)) if roi_gray.size > 0 else mean_val

    roi_contrast_ratio = float(roi_mean / (mean_val + 1.0))
    brightness_ratio = float(mean_val / max(state.long_term_brightness, 1.0))
    brightness_change = float((mean_val - prev_mean) / max(prev_mean, 1.0))

    gray_display = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    opening_mask = cv2.bitwise_or(portal_mask_full, light_mask_full)
    opening_mask = cv2.bitwise_or(opening_mask, ceiling_structure_mask_full)
    opening_display = cv2.cvtColor(opening_mask, cv2.COLOR_GRAY2BGR)

    overlay_mask = np.zeros_like(color_frame)
    overlay_mask[portal_mask_full > 0] = (255, 180, 0)
    overlay_mask[light_mask_full > 0] = (0, 255, 255)
    overlay_mask[ceiling_structure_mask_full > 0] = (255, 255, 255)
    overlay_view = cv2.addWeighted(color_frame, 0.78, overlay_mask, 0.55, 0)

    return {
        "mean": mean_val,
        "std": std_val,
        "texture_score": texture_score,
        "upper_texture_score": upper_texture_score,
        "center_texture_score": center_texture_score,
        "brightness_ratio": brightness_ratio,
        "brightness_change": brightness_change,
        "roi_contrast": roi_contrast_ratio,
        "is_locked": arch_presence,
        "long_horizontal_count": 0,
        "total_long_length": 0.0,
        "sky_edge_density": sky_edge_density,
        "edge_density": edge_density,
        "saturation": saturation,
        "dark_growth_ema": state.dark_growth_ema,
        "bright_growth_ema": state.bright_growth_ema,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "tunnel_light_total_count": tunnel_light_total_count,
        "tunnel_light_side_count": tunnel_light_side_count,
        "tunnel_light_alignment_score": tunnel_light_alignment_score,
        "tunnel_light_alignment_residual": tunnel_light_alignment_residual,
        "tunnel_light_gap_cv": tunnel_light_gap_cv,
        "tunnel_light_band_score": tunnel_light_band_score,
        "tunnel_light_y_mean": tunnel_light_y_mean,
        "tunnel_light_presence": tunnel_light_presence,
        "ceiling_structure_stripe_count": ceiling_structure_stripe_count,
        "ceiling_structure_tall_count": ceiling_structure_tall_count,
        "ceiling_structure_left_count": ceiling_structure_left_count,
        "ceiling_structure_right_count": ceiling_structure_right_count,
        "ceiling_structure_area": ceiling_structure_area,
        "ceiling_structure_max_height": ceiling_structure_max_height,
        "ceiling_structure_presence": ceiling_structure_presence,
        "portal_arch_score": portal_arch_score,
        "portal_area_ratio": portal_area_ratio,
        "portal_coverage": portal_coverage,
        "portal_depth": portal_depth,
        "portal_center_offset": portal_center_offset,
        "portal_width_ratio": portal_width_ratio,
        "portal_reflector_segment_count": portal_reflector_segment_count,
        "portal_reflector_coverage": portal_reflector_coverage,
        "arch_presence": arch_presence,
        "processed_views": {
            "gray": gray_display,
            "canny": edges_display,
            "opening": opening_display,
            "morph": overlay_view,
        },
    }


def recognize_scene_raw(features, thresholds):
    recent_avg = np.mean(state.brightness_history[-6:]) if state.brightness_history else features["mean"]
    outside_scene = (
        "白天"
        if (
            recent_avg >= thresholds["outside_day_mean"]
            or (
                state.environment == "白天"
                and recent_avg >= thresholds["outside_day_hysteresis"]
            )
        )
        else "黑夜"
    )
    state.environment = outside_scene

    # 隧道内的核心证据只保留三类：
    # 1. 上方规律灯带
    # 2. 顶部纵向/弯曲长条结构
    # 3. 靠近洞口时的拱形暗区
    soft_light_total_min = max(4, thresholds["regular_light_total_min"] - 2)
    soft_light_side_min = max(2, thresholds["regular_light_side_min"] - 1)
    soft_light_band_score_min = max(0.90, thresholds["regular_light_band_score_min"] - 0.75)
    soft_light_gap_cv_max = thresholds["regular_light_gap_cv_max"] + 0.35
    soft_light_y_min = max(0.15, thresholds["regular_light_y_min"] - 0.35)
    strong_light_y_min = max(0.18, thresholds["regular_light_y_min"] - 0.25)

    strong_light_condition = (
        features["tunnel_light_total_count"] >= thresholds["regular_light_total_min"]
        and features["tunnel_light_side_count"] >= thresholds["regular_light_side_min"]
        and features["tunnel_light_band_score"] >= thresholds["regular_light_band_score_min"]
        and features["tunnel_light_gap_cv"] <= thresholds["regular_light_gap_cv_max"] + 0.08
        and features["tunnel_light_y_mean"] >= strong_light_y_min
    )
    soft_light_condition = (
        features["tunnel_light_total_count"] >= soft_light_total_min
        and features["tunnel_light_side_count"] >= soft_light_side_min
        and features["tunnel_light_band_score"] >= soft_light_band_score_min
        and features["tunnel_light_gap_cv"] <= soft_light_gap_cv_max
        and features["tunnel_light_y_mean"] >= soft_light_y_min
    )

    structure_support_condition = (
        features["ceiling_structure_stripe_count"] >= thresholds["ceiling_structure_stripe_min"]
        and features["ceiling_structure_area"] >= thresholds["ceiling_structure_area_min"]
    )
    structure_bilateral_condition = (
        features["ceiling_structure_left_count"] >= 1
        and features["ceiling_structure_right_count"] >= 1
    )
    structure_strong_condition = (
        (
            features["ceiling_structure_stripe_count"] >= max(3, thresholds["ceiling_structure_pair_min"] + 1)
            or features["ceiling_structure_tall_count"] >= max(2, thresholds["ceiling_structure_tall_min"] + 1)
        )
        and features["ceiling_structure_area"] >= thresholds["ceiling_structure_area_min"]
    )
    arch_condition = features["arch_presence"]
    arch_entry_support_condition = (
        arch_condition
        and features["portal_width_ratio"] >= thresholds["arch_width_ratio_min"]
        and (
            features["portal_reflector_segment_count"] >= thresholds["portal_reflector_segment_min"]
            or features["portal_reflector_coverage"] >= thresholds["portal_reflector_coverage_min"]
            or features["portal_depth"] >= 0.45
        )
    )
    bootstrap_inside_condition = (
        state.frame_count <= 12
        and outside_scene == "白天"
        and features["mean"] <= 120
        and features["saturation"] >= 120
        and features["tunnel_light_total_count"] >= 6
        and features["tunnel_light_side_count"] >= 4
        and (
            features["tunnel_light_band_score"] >= 1.00
            or structure_support_condition
        )
    )
    day_direct_inside_condition = (
        outside_scene == "白天"
        and features["tunnel_light_total_count"] >= 8
        and features["tunnel_light_side_count"] >= 5
        and features["tunnel_light_band_score"] >= 1.30
        and features["roi_contrast"] <= 1.05
        and features["saturation"] <= 100
    )
    night_direct_inside_condition = (
        outside_scene == "黑夜"
        and structure_strong_condition
        and soft_light_condition
        and features["roi_contrast"] >= 1.10
    )
    portal_entry_condition = (
        arch_entry_support_condition
        and (structure_bilateral_condition or features["ceiling_structure_area"] >= 2000)
        and (
            features["brightness_ratio"] <= 1.05
            or features["roi_contrast"] <= 0.92
        )
        and (
            features["tunnel_light_total_count"] >= 4
            or features["ceiling_structure_area"] >= 2000
        )
    )
    # 白天靠近洞口时，经常先看到暗洞和反光条，这时允许它更快进入隧道候选。
    dark_portal_entry_condition = (
        outside_scene == "白天"
        and features["brightness_ratio"] <= 0.86
        and features["roi_contrast"] <= 0.80
        and features["saturation"] <= 120
        and (
            arch_entry_support_condition
            or (
                features["tunnel_light_total_count"] >= 6
                and features["tunnel_light_side_count"] >= 3
            )
        )
    )
    day_open_road_light_condition = (
        outside_scene == "白天"
        and not arch_entry_support_condition
        and not structure_bilateral_condition
        and features["mean"] <= 135
        and features["brightness_ratio"] >= 0.97
        and features["roi_contrast"] >= 0.94
        and features["saturation"] >= 100
        and features["tunnel_light_side_count"] >= 5
        and features["tunnel_light_band_score"] >= 1.35
        and features["tunnel_light_band_score"] <= 2.20
    )
    night_road_release_condition = (
        features["mean"] <= 105
        and features["brightness_ratio"] >= 1.08
        and features["saturation"] >= 125
        and features["tunnel_light_side_count"] <= 5
        and features["tunnel_light_band_score"] < 1.70
        and features["ceiling_structure_area"] < 1900
        and (
            not arch_condition
            or features["ceiling_structure_area"] < 1000
        )
    )
    # 夜路单边路灯经常会在上方打出几根长条，但它没有隧道那种双侧包裹感，
    # 也没有靠近画面上沿的规则灯带，这类画面要明确挡在洞外。
    night_single_side_road_condition = (
        outside_scene == "黑夜"
        and not arch_condition
        and not structure_bilateral_condition
        and features["ceiling_structure_left_count"] + features["ceiling_structure_right_count"] >= 4
        and features["tunnel_light_side_count"] >= 4
        and features["tunnel_light_y_mean"] <= 0.45
        and features["tunnel_light_band_score"] <= 1.10
        and features["tunnel_light_alignment_score"] <= 1.85
        and features["brightness_ratio"] >= 0.88
        and features["roi_contrast"] >= 0.72
        and features["saturation"] >= 105
    )

    # “很多长条 + 规律黄灯”是隧道内主判据；
    # 如果只抓到少量灯和一根长条，就更像是入口过渡而不是已经稳定进洞。
    inside_core_condition = (
        (
            bootstrap_inside_condition
            or day_direct_inside_condition
            or night_direct_inside_condition
            or dark_portal_entry_condition
            or
            (
                structure_strong_condition
                and soft_light_condition
                and (
                    (
                        outside_scene == "黑夜"
                        and (
                            features["roi_contrast"] <= 1.10
                            or arch_entry_support_condition
                        )
                    )
                    or (
                        outside_scene == "白天"
                        and (
                            structure_bilateral_condition
                            or arch_entry_support_condition
                        )
                        and features["roi_contrast"] <= 1.05
                    )
                )
            )
            or (
                structure_strong_condition
                and arch_entry_support_condition
                and features["brightness_ratio"] <= 1.05
            )
        )
        and not night_road_release_condition
        and not day_open_road_light_condition
    )
    entry_condition = (
        (
            structure_support_condition
            and soft_light_condition
            and not structure_strong_condition
            and features["saturation"] >= 60
        )
        or portal_entry_condition
    )

    clear_day_release_condition = (
        outside_scene == "白天"
        and (
            (
                features["roi_contrast"] >= 1.30
                and features["brightness_ratio"] >= 1.08
                and features["tunnel_light_side_count"] <= 4
            )
            or (
                features["roi_contrast"] >= 1.14
                and features["brightness_ratio"] >= 1.00
                and features["saturation"] <= 70
                and (
                    features["tunnel_light_side_count"] <= 4
                    or (
                        features["ceiling_structure_area"] < 1300
                        and features["tunnel_light_band_score"] < 1.70
                    )
                )
            )
            or (
                features["mean"] >= 130
                and features["brightness_ratio"] >= 1.00
                and features["roi_contrast"] >= 0.94
                and features["saturation"] >= 82
                and features["tunnel_light_total_count"] <= 8
                and features["tunnel_light_side_count"] <= 4
                and features["tunnel_light_band_score"] < 1.20
            )
            or day_open_road_light_condition
        )
    )

    # 已经进洞后允许短时掉点，但如果画面重新变亮、顶部长条也消失，就要尽快退出。
    light_hold_condition = (
        (
            strong_light_condition
            or (
                state.in_tunnel_session
                and outside_scene == "白天"
                and features["tunnel_light_total_count"] >= 8
                and features["tunnel_light_side_count"] >= 5
                and features["tunnel_light_band_score"] >= 1.25
            )
            or (
                state.in_tunnel_session
                and features["tunnel_light_total_count"] >= 6
                and features["tunnel_light_side_count"] >= 3
                and features["tunnel_light_band_score"] >= 1.40
                and features["brightness_ratio"] <= 1.05
            )
        )
        and not clear_day_release_condition
        and not night_road_release_condition
        and not day_open_road_light_condition
    )
    day_arch_hold_condition = (
        state.in_tunnel_session
        and outside_scene == "白天"
        and arch_condition
        and features["portal_depth"] >= thresholds["arch_depth_min"]
        and features["mean"] <= 142
        and features["saturation"] <= 90
    )
    dim_day_hold_condition = (
        state.in_tunnel_session
        and outside_scene == "白天"
        and features["roi_contrast"] <= 0.94
        and features["brightness_ratio"] <= 1.10
        and not clear_day_release_condition
        and not night_road_release_condition
        and not day_open_road_light_condition
    )
    dim_night_hold_condition = (
        state.in_tunnel_session
        and outside_scene == "黑夜"
        and features["roi_contrast"] <= 0.92
        and features["brightness_ratio"] <= 1.18
        and features["saturation"] <= 110
        and not night_road_release_condition
    )
    structure_hold_condition = (
        state.in_tunnel_session
        and (
            (
                outside_scene == "白天"
                and structure_strong_condition
                and structure_bilateral_condition
                and features["roi_contrast"] <= 1.00
                and features["brightness_ratio"] <= 1.08
                and features["saturation"] <= 120
                and not night_road_release_condition
                and not day_open_road_light_condition
            )
            or (
                outside_scene == "黑夜"
                and structure_support_condition
                and features["brightness_ratio"] <= 1.15
                and features["saturation"] <= 120
                and not night_road_release_condition
            )
        )
    )
    dark_hold_condition = (
        state.in_tunnel_session
        and outside_scene == "黑夜"
        and (structure_support_condition or arch_condition or strong_light_condition)
        and features["mean"] <= thresholds["tunnel_dark_hold_mean_max"] + 18
        and features["brightness_ratio"] <= thresholds["tunnel_dark_hold_ratio_max"] + 0.18
        and features["saturation"] <= 110
        and not night_road_release_condition
    )
    inside_hold_condition = (
        inside_core_condition
        or light_hold_condition
        or day_arch_hold_condition
        or dim_day_hold_condition
        or dim_night_hold_condition
        or structure_hold_condition
        or dark_hold_condition
    )
    if night_road_release_condition or day_open_road_light_condition:
        inside_hold_condition = False

    confidence = 42
    if strong_light_condition:
        confidence += 20
    elif soft_light_condition:
        confidence += 12
    if structure_strong_condition:
        confidence += 24
    elif structure_support_condition:
        confidence += 14
    if arch_condition:
        confidence += 8
    if inside_core_condition:
        confidence += 10
    elif outside_scene == "白天":
        confidence += 16
    else:
        confidence += 10
    state.confidence = min(99, max(35, int(confidence)))

    if not state.in_tunnel_session:
        state.tunnel_session_frames = 0
        state.outside_session_frames += 1
        min_outside_gap_frames = (
            thresholds["day_outside_min_gap_frames"]
            if outside_scene == "白天"
            else thresholds["night_outside_min_gap_frames"]
        )
        reentry_cooldown_frames = thresholds["reentry_cooldown_frames"]
        in_reentry_cooldown = (state.frame_count - state.last_exit_time) <= reentry_cooldown_frames
        cooldown_reentry_allowed = (
            bootstrap_inside_condition
            or arch_entry_support_condition
            or (
                strong_light_condition
                and structure_bilateral_condition
            )
        )
        if in_reentry_cooldown and not cooldown_reentry_allowed:
            state.entrance_detected_frames = 0
            state.exit_detected_frames = 0
            state.non_tunnel_frames = 0
            return outside_scene
        if night_single_side_road_condition:
            state.entrance_detected_frames = 0
            state.exit_detected_frames = 0
            state.non_tunnel_frames = 0
            return outside_scene
        if inside_core_condition:
            if not bootstrap_inside_condition and state.outside_session_frames < min_outside_gap_frames:
                return outside_scene
            state.entrance_detected_frames += 1
            state.exit_detected_frames = 0
            state.non_tunnel_frames = 0
            if bootstrap_inside_condition:
                state.in_tunnel_session = True
                state.entrance_detected_frames = 0
                state.event_hold_frames = max(8, thresholds["event_hold_frames"] * 3)
                state.non_tunnel_frames = 0
                state.tunnel_session_frames = 0
                state.outside_session_frames = 0
                return "隧道内"
            if state.entrance_detected_frames >= thresholds["tunnel_entry_confirm_frames"]:
                state.in_tunnel_session = True
                state.entrance_detected_frames = 0
                state.event_hold_frames = max(6, thresholds["event_hold_frames"] * 2)
                state.non_tunnel_frames = 0
                state.tunnel_session_frames = 0
                state.outside_session_frames = 0
                return "隧道内"
            if (
                outside_scene == "白天"
                and state.entrance_detected_frames >= thresholds["arch_entry_confirm_frames"]
            ):
                return "隧道入口"
            return outside_scene

        if entry_condition:
            if day_open_road_light_condition:
                state.entrance_detected_frames = 0
                return outside_scene
            if state.outside_session_frames < min_outside_gap_frames:
                return outside_scene
            state.entrance_detected_frames = min(
                state.entrance_detected_frames + 1,
                thresholds["arch_entry_confirm_frames"],
            )
            state.exit_detected_frames = 0
            state.non_tunnel_frames = 0
            if state.entrance_detected_frames >= thresholds["arch_entry_confirm_frames"]:
                state.in_tunnel_session = True
                state.event_hold_frames = max(8, thresholds["event_hold_frames"] * 3)
                state.entrance_detected_frames = 0
                state.tunnel_session_frames = 0
                state.outside_session_frames = 0
                return "隧道入口"
            return outside_scene

        state.entrance_detected_frames = 0
        state.exit_detected_frames = 0
        state.non_tunnel_frames = 0
        return outside_scene

    state.tunnel_session_frames += 1
    state.outside_session_frames = 0
    if inside_hold_condition:
        state.exit_detected_frames = 0
        state.non_tunnel_frames = 0
        return "隧道内"

    if state.event_hold_frames > 0:
        state.event_hold_frames -= 1
        state.exit_detected_frames = 0
        state.non_tunnel_frames = 0
        return "隧道内"

    state.non_tunnel_frames += 1
    clear_day_exit = (
        day_open_road_light_condition
        or (
            clear_day_release_condition
            and (
                not soft_light_condition
                or features["brightness_ratio"] >= thresholds["tunnel_exit_brightness_ratio_min"]
                or features["roi_contrast"] >= thresholds["tunnel_exit_roi_min"]
            )
        )
    )

    if outside_scene == "白天":
        day_release_support = (
            (
                state.non_tunnel_frames >= thresholds["tunnel_support_release_frames"]
                and features["roi_contrast"] >= 0.94
                and features["brightness_ratio"] >= 0.98
                and features["tunnel_light_side_count"] <= 4
                and features["tunnel_light_band_score"] < 1.10
            )
            or night_road_release_condition
        )
        if clear_day_exit or day_release_support:
            state.exit_detected_frames += 1
        else:
            state.exit_detected_frames = 0
        allow_day_exit = (
            state.tunnel_session_frames >= thresholds["day_tunnel_min_frames"]
            or day_open_road_light_condition
        )
        if allow_day_exit and state.exit_detected_frames >= thresholds["tunnel_day_exit_confirm_frames"]:
            state.in_tunnel_session = False
            state.exit_detected_frames = 0
            state.entrance_detected_frames = 0
            state.non_tunnel_frames = 0
            state.tunnel_session_frames = 0
            state.outside_session_frames = 0
            state.last_exit_time = state.frame_count
            return outside_scene

        if allow_day_exit and state.exit_detected_frames >= max(1, thresholds["tunnel_day_exit_confirm_frames"] - 1):
            return "隧道出口"
        return "隧道内"

    clear_night_exit = (
        (
            outside_scene == "黑夜"
            and (
                (
                    features["brightness_ratio"] >= 1.08
                    and features["saturation"] >= 110
                    and features["tunnel_light_band_score"] < 1.80
                    and features["tunnel_light_side_count"] <= 5
                    and features["ceiling_structure_area"] < 1400
                )
                or (
                    features["brightness_ratio"] >= 1.22
                    and features["saturation"] >= 140
                    and features["tunnel_light_side_count"] <= 3
                )
            )
        )
        or night_road_release_condition
    )
    if clear_night_exit:
        state.exit_detected_frames += 1
    else:
        state.exit_detected_frames = 0

    strong_night_release = (
        state.exit_detected_frames >= thresholds["tunnel_night_exit_confirm_frames"] + 2
    )
    if (
        (
            state.tunnel_session_frames >= thresholds["night_tunnel_min_frames"]
            and state.exit_detected_frames >= thresholds["tunnel_night_exit_confirm_frames"]
        )
        or strong_night_release
    ):
        state.in_tunnel_session = False
        state.exit_detected_frames = 0
        state.entrance_detected_frames = 0
        state.non_tunnel_frames = 0
        state.tunnel_session_frames = 0
        state.outside_session_frames = 0
        state.last_exit_time = state.frame_count
        return outside_scene

    if state.exit_detected_frames >= max(1, thresholds["tunnel_night_exit_confirm_frames"] - 1):
        return "隧道出口"
    return "隧道内"


def get_smoothed_scene(raw_scene, thresholds):
    base_raw_scene = raw_scene
    if raw_scene == "隧道入口":
        base_raw_scene = "隧道内"
    elif raw_scene == "隧道出口":
        base_raw_scene = state.environment if state.environment in {"白天", "黑夜"} else state.post_exit_scene

    if state.frame_count <= 1:
        state.current_base_scene = base_raw_scene
        state.current_stable_scene = base_raw_scene
        state.pending_scene = None
        state.stable_transition_frames = 0
        state.entrance_pending_display = False
        state.exit_pending_display = False
        return state.current_stable_scene

    if state.entrance_pending_display:
        if state.current_base_scene != "隧道内":
            state.entrance_pending_display = False
            state.entrance_display_counter = 0
            state.current_stable_scene = state.current_base_scene
            return state.current_stable_scene
        if state.entrance_display_counter > 0:
            state.entrance_display_counter -= 1
            return "隧道入口"
        state.entrance_pending_display = False
        state.entrance_display_counter = 0
        state.current_stable_scene = state.current_base_scene
        return state.current_stable_scene

    if state.exit_pending_display:
        if state.current_base_scene == "隧道内":
            state.exit_pending_display = False
            state.exit_display_counter = 0
            state.current_stable_scene = "隧道内"
            return state.current_stable_scene
        if state.exit_display_counter > 0:
            state.exit_display_counter -= 1
            return "隧道出口"
        state.exit_pending_display = False
        state.exit_display_counter = 0
        state.current_stable_scene = state.current_base_scene
        return state.current_stable_scene

    if base_raw_scene == state.current_base_scene:
        state.pending_scene = None
        state.stable_transition_frames = 0
        state.current_stable_scene = state.current_base_scene
        return state.current_stable_scene

    if state.pending_scene == base_raw_scene:
        state.stable_transition_frames += 1
    else:
        state.pending_scene = base_raw_scene
        state.stable_transition_frames = 1

    if state.current_base_scene == "隧道内" and base_raw_scene in {"白天", "黑夜"}:
        required_frames = 2
    elif state.current_base_scene in {"白天", "黑夜"} and base_raw_scene == "隧道内":
        required_frames = 2
    else:
        required_frames = 2

    if state.stable_transition_frames >= required_frames:
        previous_base_scene = state.current_base_scene
        state.current_base_scene = base_raw_scene
        state.pending_scene = None
        state.stable_transition_frames = 0
        transition_display_counter = max(
            1,
            int(thresholds.get("transition_display_frames", thresholds["event_hold_frames"])) - 1,
        )

        if previous_base_scene in {"白天", "黑夜"} and state.current_base_scene == "隧道内":
            state.pre_entry_scene = previous_base_scene
            state.current_stable_scene = "隧道入口"
            state.entrance_pending_display = True
            state.entrance_display_counter = transition_display_counter
            state.exit_pending_display = False
            return state.current_stable_scene

        if previous_base_scene == "隧道内" and state.current_base_scene in {"白天", "黑夜"}:
            state.post_exit_scene = state.current_base_scene
            state.current_stable_scene = "隧道出口"
            state.exit_pending_display = True
            state.exit_display_counter = transition_display_counter
            state.entrance_pending_display = False
            return state.current_stable_scene

        state.current_stable_scene = state.current_base_scene
        return state.current_stable_scene

    state.current_stable_scene = state.current_base_scene
    return state.current_stable_scene


def control_lights(scene):
    # 按你最新要求简化：
    # 1. 只要判成隧道内，就开近光灯
    # 2. 白天不开灯
    # 3. 黑夜开远光灯
    if scene == "隧道内":
        return {"low_beam": "开启", "high_beam": "关闭"}
    if scene == "黑夜":
        return {"low_beam": "关闭", "high_beam": "开启"}
    return {"low_beam": "关闭", "high_beam": "关闭"}


def validate_feedback(scene, light_status):
    low_beam = light_status["low_beam"]
    high_beam = light_status["high_beam"]

    if scene == "隧道内" and (low_beam != "开启" or high_beam != "关闭"):
        return "未达标"
    if scene == "黑夜" and (low_beam != "关闭" or high_beam != "开启"):
        return "未达标"
    if scene in {"白天", "隧道入口", "隧道出口"} and (low_beam != "关闭" or high_beam != "关闭"):
        return "未达标"
    return "达标"


def _write_log(stable_scene, raw_scene, light_status, validation, features):
    # TXT 按“每帧一行”写，后面老师或自己复盘时直接翻就能看每帧怎么判的。
    log_line = (
        f"{state.frame_count:4d},"
        f"{state.environment},"
        f"{raw_scene},"
        f"{stable_scene},"
        f"{light_status['low_beam']},"
        f"{validation},"
        f"{features['mean']:.1f},"
        f"{features['brightness_ratio']:.3f},"
        f"{features['brightness_change']:.3f},"
        f"{features['texture_score']:.2f},"
        f"{features['upper_texture_score']:.2f},"
        f"{features['center_texture_score']:.2f},"
        f"{features['edge_density']:.4f},"
        f"{features['sky_edge_density']:.4f},"
        f"{features['roi_contrast']:.3f},"
        f"{features['saturation']:.1f},"
        f"{features['long_horizontal_count']},"
        f"{features['total_long_length']:.1f},"
        f"{features['tunnel_light_total_count']},"
        f"{features['tunnel_light_side_count']},"
        f"{features['tunnel_light_alignment_score']:.2f},"
        f"{features['tunnel_light_band_score']:.2f},"
        f"{0.0 if features['tunnel_light_gap_cv'] > 900 else features['tunnel_light_gap_cv']:.3f},"
        f"{features['tunnel_light_y_mean']:.3f},"
        f"{int(features['tunnel_light_presence'])},"
        f"{features['ceiling_structure_stripe_count']},"
        f"{features['ceiling_structure_tall_count']},"
        f"{features['ceiling_structure_area']:.1f},"
        f"{int(features['ceiling_structure_presence'])},"
        f"{features['portal_arch_score']:.2f},"
        f"{features['portal_area_ratio']:.3f},"
        f"{features['portal_depth']:.3f},"
        f"{int(features['arch_presence'])},"
        f"{features['dark_growth_ema']:.2f},"
        f"{features['bright_growth_ema']:.2f},"
        f"{features['canny_low']},"
        f"{features['canny_high']}\n"
    )

    if state.frame_count == 1:
        with open(LOG_FILE, "w", encoding="utf-8") as file:
            file.write(
                "帧号,环境,原始场景,平滑场景,近光灯,验证结果,平均亮度,相对亮度,亮度变化率,纹理度,上部纹理度,中心纹理度,边缘密度,天空边缘密度,ROI对比度,饱和度,长直线数量,长直线总长度,顶灯总数,单侧顶灯数,顶灯排列评分,灯带规则度,灯带间距波动,顶灯平均高度,顶灯特征触发,顶部长条数量,顶部高长条数量,顶部长条面积,顶部长条特征触发,拱形得分,拱形面积比,拱形深度,拱形特征触发,暗区增长,亮区增长,Canny低阈值,Canny高阈值\n"
            )
            file.write(log_line)
    else:
        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(log_line)


def process_frame(frame, thresholds=None):
    merged_thresholds = _merge_thresholds(thresholds)
    color_frame, gray_frame = preprocess_frame(frame)
    features = extract_features(color_frame, gray_frame, merged_thresholds)
    raw_scene = recognize_scene_raw(features, merged_thresholds)
    stable_scene = get_smoothed_scene(raw_scene, merged_thresholds)
    light_status = control_lights(stable_scene)
    validation = validate_feedback(stable_scene, light_status)

    _write_log(stable_scene, raw_scene, light_status, validation, features)

    return {
        "processed_views": features["processed_views"],
        "scene": stable_scene,
        "raw_scene": raw_scene,
        "light_status": light_status,
        "validation": validation,
        "log_file": LOG_FILE,
        "features": {
            "brightness_mean": features["mean"],
            "brightness_ratio": features["brightness_ratio"],
            "brightness_change": features["brightness_change"],
            "texture_score": features["texture_score"],
            "upper_texture_score": features["upper_texture_score"],
            "center_texture_score": features["center_texture_score"],
            "edge_density": features["edge_density"],
            "sky_edge_density": features["sky_edge_density"],
            "dark_growth_ema": features["dark_growth_ema"],
            "bright_growth_ema": features["bright_growth_ema"],
            "is_locked": features["is_locked"],
            "confidence": state.confidence,
            "long_horizontal_count": features["long_horizontal_count"],
            "total_long_length": features["total_long_length"],
            "roi_contrast": features["roi_contrast"],
            "saturation": features["saturation"],
            "tunnel_light_total_count": features["tunnel_light_total_count"],
            "tunnel_light_side_count": features["tunnel_light_side_count"],
            "tunnel_light_alignment_score": features["tunnel_light_alignment_score"],
            "tunnel_light_band_score": features["tunnel_light_band_score"],
            "tunnel_light_gap_cv": features["tunnel_light_gap_cv"],
            "tunnel_light_y_mean": features["tunnel_light_y_mean"],
            "tunnel_light_presence": features["tunnel_light_presence"],
            "ceiling_structure_stripe_count": features["ceiling_structure_stripe_count"],
            "ceiling_structure_tall_count": features["ceiling_structure_tall_count"],
            "ceiling_structure_left_count": features["ceiling_structure_left_count"],
            "ceiling_structure_right_count": features["ceiling_structure_right_count"],
            "ceiling_structure_area": features["ceiling_structure_area"],
            "ceiling_structure_presence": features["ceiling_structure_presence"],
            "portal_arch_score": features["portal_arch_score"],
            "portal_depth": features["portal_depth"],
            "portal_reflector_segment_count": features["portal_reflector_segment_count"],
            "portal_reflector_coverage": features["portal_reflector_coverage"],
            "arch_presence": features["arch_presence"],
        },
    }
