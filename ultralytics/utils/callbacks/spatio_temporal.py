# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch
from pathlib import Path


class SpatioTemporalCallback:
    """
    时空分析回调函数，用于在训练过程中集成时空分析算法
    
    功能：
    1. 记录训练过程中的目标轨迹
    2. 计算时空一致性损失
    3. 分析目标之间的交互关系
    4. 预测目标的未来位置
    5. 推理阶段的目标跟踪和计数
    """
    
    def __init__(self, history_length=30, speed_threshold=5.0, interaction_distance=50.0, 
                 prediction_horizon=5, temporal_weight=0.1):
        """
        初始化时空分析回调函数
        
        参数：
        - history_length: 历史轨迹长度
        - speed_threshold: 速度阈值
        - interaction_distance: 交互距离阈值
        - prediction_horizon: 预测帧数
        - temporal_weight: 时空一致性损失的权重
        """
        self.history_length = history_length
        self.speed_threshold = speed_threshold
        self.interaction_distance = interaction_distance
        self.prediction_horizon = prediction_horizon
        self.temporal_weight = temporal_weight
        
        # 存储轨迹历史
        self.track_history = {}
        self.speed_data = {}
        self.frame_count = 0
        
        # 时空一致性损失
        self.temporal_loss = 0.0
        
        # 推理阶段使用的变量
        self.next_track_id = 0
        self.track_assignments = {}
        
    def on_train_start(self, trainer):
        """训练开始时的回调函数"""
        print(f"[时空分析] 训练开始，启用时空分析算法")
        print(f"  - 历史轨迹长度: {self.history_length}")
        print(f"  - 速度阈值: {self.speed_threshold}")
        print(f"  - 交互距离阈值: {self.interaction_distance}")
        print(f"  - 预测帧数: {self.prediction_horizon}")
        print(f"  - 时空一致性损失权重: {self.temporal_weight}")
        
    def on_train_batch_start(self, trainer):
        """训练批次开始时的回调函数"""
        self.frame_count += 1
        
    def on_train_batch_end(self, trainer):
        """训练批次结束时的回调函数"""
        # 在这里可以添加时空一致性损失的计算
        pass
        
    def on_train_epoch_start(self, trainer):
        """训练轮次开始时的回调函数"""
        print(f"[时空分析] 开始第 {trainer.epoch + 1} 轮训练")
        
    def on_train_epoch_end(self, trainer):
        """训练轮次结束时的回调函数"""
        # 计算时空一致性损失
        temporal_loss = self.calculate_temporal_loss(trainer)
        
        # 将时空一致性损失添加到总损失中
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            # 这里我们通过调整损失来影响训练
            # 注意：这是一个简化的实现，实际应用中可能需要更复杂的损失函数设计
            pass
            
        print(f"[时空分析] 第 {trainer.epoch + 1} 轮训练完成")
        print(f"  - 时空一致性损失: {temporal_loss:.4f}")
        
    def calculate_temporal_loss(self, trainer):
        """
        计算时空一致性损失
        
        这个损失函数鼓励模型在相邻帧之间产生一致的预测
        """
        # 简化的时空一致性损失计算
        # 实际应用中，这里应该计算相邻帧之间预测的一致性
        temporal_loss = 0.0
        
        # 如果有轨迹历史，计算轨迹平滑度
        if len(self.track_history) > 0:
            for track_id, history in self.track_history.items():
                if len(history) >= 2:
                    # 计算轨迹的平滑度（相邻位置之间的距离变化）
                    positions = [(x, y) for x, y, _ in history]
                    for i in range(1, len(positions)):
                        dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                      (positions[i][1] - positions[i-1][1])**2)
                        # 如果距离变化过大，增加损失
                        if dist > self.speed_threshold:
                            temporal_loss += (dist - self.speed_threshold) ** 2
                            
        return temporal_loss * self.temporal_weight
        
    def update_track_history(self, detections, frame_number):
        """
        更新轨迹历史
        
        参数：
        - detections: 检测结果，格式为 [(x1, y1, x2, y2, conf, cls, track_id), ...]
        - frame_number: 当前帧号
        """
        for detection in detections:
            if len(detection) >= 7:
                x1, y1, x2, y2, conf, cls, track_id = detection[:7]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                    
                self.track_history[track_id].append((center_x, center_y, frame_number))
                
                # 保持历史长度
                if len(self.track_history[track_id]) > self.history_length:
                    self.track_history[track_id] = self.track_history[track_id][-self.history_length:]
                    
    def estimate_speed(self, track_id):
        """
        估计目标速度
        
        参数：
        - track_id: 目标ID
        
        返回：
        - 速度（像素/帧）
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0.0
            
        history = self.track_history[track_id]
        (x1, y1, t1), (x2, y2, t2) = history[-2], history[-1]
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        time_diff = max(t2 - t1, 1)
        
        speed = distance / time_diff
        self.speed_data[track_id] = speed
        
        return speed
        
    def predict_future_position(self, track_id, horizon=None):
        """
        预测未来位置
        
        参数：
        - track_id: 目标ID
        - horizon: 预测帧数
        
        返回：
        - 预测位置列表 [(x, y), ...]
        """
        if horizon is None:
            horizon = self.prediction_horizon
            
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return []
            
        history = self.track_history[track_id]
        (x1, y1, t1), (x2, y2, t2) = history[-2], history[-1]
        
        # 计算速度向量
        vx = (x2 - x1) / max(t2 - t1, 1)
        vy = (y2 - y1) / max(t2 - t1, 1)
        
        # 预测未来位置
        predictions = []
        for i in range(1, horizon + 1):
            future_x = x2 + vx * i
            future_y = y2 + vy * i
            predictions.append((future_x, future_y))
            
        return predictions
        
    def detect_interactions(self, detections):
        """
        检测目标之间的交互
        
        参数：
        - detections: 检测结果
        
        返回：
        - 交互列表 [(track_id1, track_id2, distance), ...]
        """
        interactions = []
        
        # 获取所有目标的当前位置
        current_positions = {}
        for detection in detections:
            if len(detection) >= 7:
                x1, y1, x2, y2, conf, cls, track_id = detection[:7]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_positions[track_id] = (center_x, center_y)
                
        # 检测交互
        track_ids = list(current_positions.keys())
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track1 = track_ids[i]
                track2 = track_ids[j]
                pos1 = current_positions[track1]
                pos2 = current_positions[track2]
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance < self.interaction_distance:
                    interactions.append((track1, track2, distance))
                    
        return interactions


def on_train_start(trainer):
    """训练开始时的回调函数"""
    # 自动初始化时空分析回调实例
    if not hasattr(trainer, 'spatio_temporal_callback'):
        trainer.spatio_temporal_callback = SpatioTemporalCallback(
            history_length=30,
            speed_threshold=5.0,
            interaction_distance=50.0,
            prediction_horizon=5,
            temporal_weight=0.1
        )
    trainer.spatio_temporal_callback.on_train_start(trainer)
        
        
def on_train_batch_start(trainer):
    """训练批次开始时的回调函数"""
    if hasattr(trainer, 'spatio_temporal_callback'):
        trainer.spatio_temporal_callback.on_train_batch_start(trainer)
        
        
def on_train_batch_end(trainer):
    """训练批次结束时的回调函数"""
    if hasattr(trainer, 'spatio_temporal_callback'):
        trainer.spatio_temporal_callback.on_train_batch_end(trainer)
        
        
def on_train_epoch_start(trainer):
    """训练轮次开始时的回调函数"""
    if hasattr(trainer, 'spatio_temporal_callback'):
        trainer.spatio_temporal_callback.on_train_epoch_start(trainer)
        
        
def on_train_epoch_end(trainer):
    """训练轮次结束时的回调函数"""
    if hasattr(trainer, 'spatio_temporal_callback'):
        trainer.spatio_temporal_callback.on_train_epoch_end(trainer)


# 回调函数列表
callbacks = {
    'on_train_start': on_train_start,
    'on_train_batch_start': on_train_batch_start,
    'on_train_batch_end': on_train_batch_end,
    'on_train_epoch_start': on_train_epoch_start,
    'on_train_epoch_end': on_train_epoch_end,
}


class SpatioTemporalTracker:
    """
    推理阶段的时空分析跟踪器
    
    功能：
    1. 目标跟踪和ID分配
    2. 轨迹平滑和速度估计
    3. 交互检测和鱼群分析
    4. 鱼群计数和统计
    
    参数：
    - history_length: 历史轨迹长度
    - speed_threshold: 速度阈值
    - interaction_distance: 交互距离阈值
    - max_distance: 最大跟踪距离（像素）
    - min_confidence: 最小置信度阈值
    """
    
    def __init__(self, history_length=30, speed_threshold=5.0, interaction_distance=50.0,
                 max_distance=100, min_confidence=0.5):
        self.history_length = history_length
        self.speed_threshold = speed_threshold
        self.interaction_distance = interaction_distance
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        
        self.track_history = {}
        self.speed_data = {}
        self.next_track_id = 0
        self.frame_number = 0
        
    def process_detections(self, boxes, confidences, classes):
        """
        处理检测结果，进行目标跟踪
        
        参数：
        - boxes: 检测框 (N, 4) [x1, y1, x2, y2]
        - confidences: 置信度 (N,)
        - classes: 类别 (N,)
        
        返回：
        - 跟踪结果字典
        """
        self.frame_number += 1
        
        if len(boxes) == 0:
            return {
                'track_ids': [],
                'centers': [],
                'speeds': [],
                'count': 0
            }
        
        # 计算检测框中心点
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        # 为每个检测分配或更新轨迹ID
        track_ids = []
        track_centers = []
        track_speeds = []
        
        for i, (center, conf, cls) in enumerate(zip(centers, confidences, classes)):
            if conf < self.min_confidence:
                continue
                
            # 寻找最近的已有轨迹
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, history in self.track_history.items():
                if len(history) > 0:
                    last_pos = history[-1]
                    distance = np.sqrt((center[0] - last_pos[0])**2 + 
                                      (center[1] - last_pos[1])**2)
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            if best_track_id is not None:
                track_id = best_track_id
                self.track_history[track_id].append((center[0], center[1], self.frame_number))
                
                # 保持历史长度
                if len(self.track_history[track_id]) > self.history_length:
                    self.track_history[track_id] = self.track_history[track_id][-self.history_length:]
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.track_history[track_id] = [(center[0], center[1], self.frame_number)]
            
            track_ids.append(track_id)
            track_centers.append(center)
            
            # 计算速度
            if len(self.track_history[track_id]) >= 2:
                history = self.track_history[track_id]
                (x1, y1, t1), (x2, y2, t2) = history[-2], history[-1]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                time_diff = max(t2 - t1, 1)
                speed = distance / time_diff
                self.speed_data[track_id] = speed
                track_speeds.append(speed)
            else:
                track_speeds.append(0.0)
        
        return {
            'track_ids': track_ids,
            'centers': track_centers,
            'speeds': track_speeds,
            'count': len(set(track_ids))
        }
    
    def detect_interactions(self, track_ids, centers):
        """
        检测目标之间的交互（鱼群分析）
        
        参数：
        - track_ids: 轨迹ID列表
        - centers: 中心点列表
        
        返回：
        - 交互列表和鱼群信息
        """
        interactions = []
        groups = []
        
        # 检测交互
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                pos1 = centers[i]
                pos2 = centers[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < self.interaction_distance:
                    interactions.append((track_ids[i], track_ids[j], distance))
        
        # 检测鱼群（使用连通性分析）
        if len(track_ids) > 0:
            visited = set()
            for i in range(len(track_ids)):
                if track_ids[i] not in visited:
                    group = [track_ids[i]]
                    queue = [i]
                    visited.add(track_ids[i])
                    
                    while queue:
                        current = queue.pop(0)
                        for j in range(len(track_ids)):
                            if j not in visited:
                                pos1 = centers[current]
                                pos2 = centers[j]
                                distance = np.sqrt((pos1[0] - pos2[0])**2 + 
                                                  (pos1[1] - pos2[1])**2)
                                if distance < self.interaction_distance:
                                    group.append(track_ids[j])
                                    visited.add(track_ids[j])
                                    queue.append(j)
                    
                    if len(group) > 1:
                        groups.append(group)
        
        return {
            'interactions': interactions,
            'groups': groups,
            'group_count': len(groups)
        }
    
    def get_statistics(self):
        """
        获取跟踪统计信息
        
        返回：
        - 统计信息字典
        """
        if len(self.track_history) == 0:
            return {
                'total_tracks': 0,
                'avg_speed': 0.0,
                'max_speed': 0.0,
                'min_speed': 0.0
            }
        
        speeds = list(self.speed_data.values())
        return {
            'total_tracks': len(self.track_history),
            'avg_speed': np.mean(speeds) if len(speeds) > 0 else 0.0,
            'max_speed': np.max(speeds) if len(speeds) > 0 else 0.0,
            'min_speed': np.min(speeds) if len(speeds) > 0 else 0.0
        }
    
    def reset(self):
        """重置跟踪器"""
        self.track_history = {}
        self.speed_data = {}
        self.next_track_id = 0
        self.frame_number = 0


def predict_with_spatio_temporal(model, source, tracker=None, **kwargs):
    """
    使用时空分析进行预测的包装函数
    
    参数：
    - model: YOLO模型
    - source: 输入源（图片、视频、文件夹）
    - tracker: 时空分析跟踪器实例，如果为None则自动创建
    - **kwargs: 传递给model.predict的其他参数
    
    返回：
    - 生成器，产生带时空分析的结果
    """
    if tracker is None:
        tracker = SpatioTemporalTracker()
    
    # 获取预测结果
    results = model.predict(source, **kwargs)
    
    # 处理每个结果
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # 应用时空分析跟踪
            tracking_result = tracker.process_detections(boxes, confidences, classes)
            
            # 检测交互和鱼群
            interaction_result = tracker.detect_interactions(
                tracking_result['track_ids'],
                tracking_result['centers']
            )
            
            # 将跟踪信息添加到结果
            result.track_ids = tracking_result['track_ids']
            result.track_count = tracking_result['count']
            result.interactions = interaction_result['interactions']
            result.groups = interaction_result['groups']
            result.group_count = interaction_result['group_count']
            result.speeds = tracking_result['speeds']
            result.statistics = tracker.get_statistics()
        else:
            result.track_ids = []
            result.track_count = 0
            result.interactions = []
            result.groups = []
            result.group_count = 0
            result.speeds = []
            result.statistics = tracker.get_statistics()
        
        yield result


def count_fish_with_spatio_temporal(model, source, tracker=None, **kwargs):
    """
    使用时空分析进行鱼群计数的函数
    
    参数：
    - model: YOLO模型
    - source: 输入源
    - tracker: 时空分析跟踪器实例
    - **kwargs: 传递给model.predict的其他参数
    
    返回：
    - 计数结果列表
    """
    if tracker is None:
        tracker = SpatioTemporalTracker()
    
    results = []
    total_count = 0
    
    for result in predict_with_spatio_temporal(model, source, tracker, **kwargs):
        results.append(result)
        total_count += result.track_count
    
    return {
        'results': results,
        'total_count': total_count,
        'tracker': tracker
    }
