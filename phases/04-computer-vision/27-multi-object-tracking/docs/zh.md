# Multi-Object Tracking & Video Memory

> Tracking is detection plus association. Detect every frame. Match this frame's detections to last frame's tracks by ID.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (YOLO Detection), Phase 4 Lesson 08 (Mask R-CNN), Phase 4 Lesson 24 (SAM 3)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish tracking-by-detection from query-based tracking and name the algorithm families (SORT, DeepSORT, ByteTrack, BoT-SORT, SAM 2 memory tracker, SAM 3.1 Object Multiplex)
- Implement IoU + Hungarian assignment from scratch for classic tracking-by-detection
- Explain SAM 2's memory bank and why it handles occlusion better than IoU-based association
- Read the three tracking metrics (MOTA, IDF1, HOTA) and pick which one matters for a given use case

## The Problem

A detector tells you where the objects are in a single frame. A tracker tells you which detection in frame `t` is the same object as a detection in frame `t-1`. Without that, you cannot count objects crossing a line, follow a ball through an occlusion, or know "car #4 has been in the lane for 8 seconds."

Tracking is essential to every video-facing product: sports analytics, surveillance, autonomous driving, medical video analysis, wildlife monitoring, wordmark counting. The core building blocks are shared: a per-frame detector, a motion model (Kalman filter or something richer), an association step (Hungarian algorithm on IoU / cosine / learned features), and a track lifecycle (birth, update, death).

2026 brought two new patterns: **SAM 2 memory-based tracking** (feature-memory instead of motion-model association) and **SAM 3.1 Object Multiplex** (shared memory for many instances of the same concept). This lesson walks the classical stack first, then the memory-based approach.

## The Concept

### Tracking-by-detection

```mermaid
flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a
```

Every tracker you will encounter in 2026 is a variation on this loop. The differences:

- **SORT** (2016): Kalman filter + IoU Hungarian. Simple, fast, no appearance model.
- **DeepSORT** (2017): SORT + a CNN-based appearance feature per track (ReID embedding). Handles crossings better.
- **ByteTrack** (2021): associates low-confidence detections as a second stage; no appearance features needed but top performer on MOT17.
- **BoT-SORT** (2022): Byte + camera motion compensation + ReID.
- **StrongSORT / OC-SORT** — ByteTrack descendants with better motion and appearance.

### Kalman filter in one paragraph

A Kalman filter maintains a per-track state `(x, y, w, h, dx, dy, dw, dh)` with a covariance. At each frame, **predict** the state using a constant-velocity model, then **update** with the matched detection. The update trusts the detection more when the predict uncertainty is high. This gives smooth trajectories and the ability to continue a track through a short occlusion (1-5 frames).

Every classical tracker uses a Kalman filter in the motion-prediction step.

### The Hungarian algorithm

Given a `M x N` cost matrix (tracks x detections), find the one-to-one assignment that minimises total cost. Cost is usually `1 - IoU(track_bbox, detection_bbox)` or negative cosine similarity of appearance features. Runtime is O((M+N)^3); for M, N up to ~1000 it is fast enough in Python via `scipy.optimize.linear_sum_assignment`.

### ByteTrack's key idea

Standard trackers drop low-confidence detections (< 0.5). ByteTrack keeps them around as **second-stage candidates**: after matching tracks to high-confidence detections, unmatched tracks try to match low-confidence detections with a slightly looser IoU threshold. Recovers short occlusions, ID switches near crowds.

### SAM 2 memory-based tracking

SAM 2 handles video by keeping a **memory bank** of per-instance spatio-temporal features. Given a prompt (click, box, text) on one frame, it encodes the instance into memory. On subsequent frames, the memory is cross-attended against the new frame's features, and the decoder produces a mask for the same instance in the new frame.

No Kalman filter, no Hungarian assignment. The association is implicit in the memory-attention operation.

Pros:
- Robust to large occlusions (memory carries instance identity across many frames).
- Open-vocabulary when combined with SAM 3's text prompts.
- Works without a separate motion model.

Cons:
- Slower than ByteTrack for many-object tracking.
- Memory bank grows; limits the context window.

### SAM 3.1 Object Multiplex

Prior SAM 2 / SAM 3 tracking keeps a separate memory bank per instance. For 50 objects, 50 memory banks. Object Multiplex (March 2026) collapses them into one shared memory with **per-instance query tokens**. Cost scales sub-linearly in number of instances.

Multiplex is the new default for crowd tracking in 2026: concert crowds, warehouse workers, traffic intersections.

### Three metrics to know

- **MOTA (Multi-Object Tracking Accuracy)** — 1 - (FN + FP + ID switches) / GT. Weighted by error type; a single metric that conflates detection and association failures.
- **IDF1 (ID F1)** — harmonic mean of ID precision and recall. Focuses specifically on how well each ground-truth track keeps its ID over time. Better than MOTA for ID-switch-sensitive tasks.
- **HOTA (Higher Order Tracking Accuracy)** — decomposes into detection accuracy (DetA) and association accuracy (AssA). The community standard since 2020; most comprehensive.

For surveillance (who is who): IDF1 is what you report. For sports analytics (counting passes): HOTA. For general academic comparison: HOTA.

## Build It

### Step 1: IoU-based cost matrix

```python
import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

### Step 2: Minimal SORT-style tracker

Fixed constant-velocity Kalman omitted for brevity — we use a simple IoU association here; in production the Kalman predict is essential. The `sort` Python package provides the full version.

```python
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]
```

60 lines. Takes per-frame detections, returns per-frame track IDs. Real systems add the Kalman predict, ByteTrack's second-stage re-match, and appearance features.

### Step 3: Synthetic trajectory test

```python
def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)
```

Three objects moving in straight lines should keep their IDs across all 20 frames.

### Step 4: ID-switch metric

```python
def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches
```

这是一个简化的 IDF1-adjacent 指标：计算真实对象更改其分配的预测轨迹 ID 的次数。真正的 MOTA / IDF1 / HOTA 工具存在于 `py-motmetrics` 和 `TrackEval` 中。

## 使用它

2026 年产量追踪：

- `ultralytics` — YOLOv8 + ByteTrack / BoT-SORT 内置。 `结果 = model.track(source, tracker="bytetrack.yaml")`。默认。
- `supervision` (Roboflow) — ByteTrack 包装器和注释实用程序。
- SAM 2 / SAM 3.1 — 通过“processor.track()”进行基于内存的跟踪。
- 自定义堆栈：检测器 (YOLOv8 / RT-DETR) + `sort-tracker` / `OC-SORT` / `StrongSORT`。

采摘：

- 行人/汽车/箱子的速度超过 30 fps：**ByteTrack with ultralytics**。
- 人群中某一类的许多实例：**SAM 3.1 对象多路复用**。
- 具有可识别外观的重度遮挡：**DeepSORT / StrongSORT**（ReID 功能）。
- 运动/复杂交互：**BoT-SORT** 或学习跟踪器 (MOTRv3)。

## 发货

本课产生：

- `outputs/prompt-tracker-picker.md` — 根据场景类型、遮挡模式和延迟预算选择 SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1。
- `outputs/skill-mot-evaluator.md` — 为 MOTA / IDF1 / HOTA 针对地面真实轨迹编写完整的评估工具。

## 练习

1. **（简单）** 使用 3、10 和 30 个对象运行上面的合成跟踪器。报告每种情况下的 ID 转换计数。确定简单的仅 IoU 关联从哪里开始失败。
2. **（中）** 在关联之前添加恒速卡尔曼预测步骤。表明短（2-3 帧）遮挡不再导致 ID 切换。
3. **（困难）** 集成 SAM 2 基于内存的跟踪器（通过“变压器”）作为替代跟踪器后端。在 30 秒的人群片段上运行 SimpleTracker 和 SAM 2，比较 ID 转换计数，手动标记 5 个显着人物的真实 ID。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|检测跟踪 | “检测然后关联” |每帧检测器 + IoU / 外观的匈牙利分配 |
|卡尔曼滤波器| “运动预测”|线性动力学 + 协方差用于平滑轨迹预测和遮挡处理 |
|匈牙利算法 | “最优分配” |解决最小成本二分匹配问题； `scipy.optimize.线性总和分配` |
|字节追踪 | “低置信度第二关”|将不匹配的轨迹与低置信度检测重新匹配以恢复短遮挡 |
|深度排序| “排序+外观”|增加ReID功能，用于跨帧匹配；更好地保存身份证件 |
|记忆库| “SAM 2 伎俩”|跨帧存储的每个实例的时空特征；交叉注意力取代显式关联 |
|对象复用| “SAM 3.1 共享内存”|具有按实例查询的单一共享内存，可实现快速多对象跟踪 |
|和田 | “现代跟踪指标” |分解为检测精度和关联精度；社区标准|

## 进一步阅读

- [SORT (Bewley et al., 2016)](https://arxiv.org/abs/1602.00763) — 最小的检测跟踪论文
- [DeepSORT (Wojke et al., 2017)](https://arxiv.org/abs/1703.07402) — 添加外观特征
- [ByteTrack (Zhang et al., 2022)](https://arxiv.org/abs/2110.06864) — 低置信度第二遍
- [BoT-SORT (Aharon et al., 2022)](https://arxiv.org/abs/2206.14651) — 相机运动补偿
- [HOTA (Luiten et al., 2020)](https://arxiv.org/abs/2009.07736) — 分解的跟踪指标
- [SAM 2 视频分割（Meta，2024）](https://ai.meta.com/sam2/) — 基于内存的跟踪器
- [SAM 3.1 对象复用（Meta，2026 年 3 月）](https://ai.meta.com/blog/segment-anything-model-3/)
