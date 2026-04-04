"""
指标定义模块
Metrics Definer Module

定义和计算优化目标指标：R2, RMSE, MAE等
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class OptimizationDirection(Enum):
    """优化方向"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    COMPOSITE_MAX = "composite_max"  # 组合指标最大化


@dataclass
class MetricConfig:
    """单个指标配置"""
    name: str
    weight: float
    mode: str  # "minimize" or "maximize"
    soft_cap: Optional[float] = None  # 对于minimize指标的上限
    soft_floor: Optional[float] = None  # 对于maximize指标的下限
    
    def to_dict(self) -> Dict:
        return {
            "weight": self.weight,
            "mode": self.mode,
            "soft_cap": self.soft_cap,
            "soft_floor": self.soft_floor
        }


class MetricsDefiner:
    """
    指标定义器
    
    负责定义、计算和聚合优化目标指标
    """
    
    # 支持的指标类型
    SUPPORTED_METRICS = {
        "r2": {"mode": "maximize", "default_weight": 0.3, "default_floor": 0.0},
        "rmse": {"mode": "minimize", "default_weight": 0.3, "default_cap": 1.0},
        "mae": {"mode": "minimize", "default_weight": 0.2, "default_cap": 0.5},
        "mse": {"mode": "minimize", "default_weight": 0.1, "default_cap": 1.0},
        "mape": {"mode": "minimize", "default_weight": 0.1, "default_cap": 100.0},
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化指标定义器
        
        Args:
            config_path: 配置文件路径
        """
        self.metrics: Dict[str, MetricConfig] = {}
        self.global_score_range = [0, 1]
        self.optim_direction = OptimizationDirection.COMPOSITE_MAX
        
        if config_path:
            self.load_config(config_path)
        else:
            self._set_default_metrics()
    
    def _set_default_metrics(self):
        """设置默认指标配置"""
        self.add_metric("r2", weight=0.3, mode="maximize", soft_floor=0.0)
        self.add_metric("rmse", weight=0.3, mode="minimize", soft_cap=1.0)
        self.add_metric("mae", weight=0.2, mode="minimize", soft_cap=0.5)
        self.add_metric("mse", weight=0.1, mode="minimize", soft_cap=1.0)
        self.add_metric("mape", weight=0.1, mode="minimize", soft_cap=100.0)
    
    def add_metric(
        self, 
        name: str, 
        weight: float, 
        mode: str,
        soft_cap: Optional[float] = None,
        soft_floor: Optional[float] = None
    ):
        """
        添加指标
        
        Args:
            name: 指标名称
            weight: 权重
            mode: 优化模式 ("minimize" or "maximize")
            soft_cap: 上限（用于minimize）
            soft_floor: 下限（用于maximize）
        """
        self.metrics[name] = MetricConfig(
            name=name,
            weight=weight,
            mode=mode,
            soft_cap=soft_cap,
            soft_floor=soft_floor
        )
    
    def load_config(self, config_path: str):
        """
        从JSON文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        self.global_score_range = config.get("global_score_range", [0, 1])
        self.optim_direction = OptimizationDirection(
            config.get("optim_direction", "composite_max")
        )
        
        for metric_name, metric_cfg in config.get("metrics_weights", {}).items():
            self.add_metric(
                name=metric_name,
                weight=metric_cfg.get("weight", 0.1),
                mode=metric_cfg.get("mode", "minimize"),
                soft_cap=metric_cfg.get("soft_cap"),
                soft_floor=metric_cfg.get("soft_floor")
            )
    
    def save_config(self, config_path: str):
        """
        保存配置到JSON文件
        
        Args:
            config_path: 配置文件路径
        """
        config = {
            "global_score_range": self.global_score_range,
            "optim_direction": self.optim_direction.value,
            "metrics_weights": {
                name: cfg.to_dict() 
                for name, cfg in self.metrics.items()
            }
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def normalize_score(self, raw_value: float, metric_config: MetricConfig) -> float:
        """
        归一化单个指标分数到 [0, 1] 范围
        
        Args:
            raw_value: 原始指标值
            metric_config: 指标配置
            
        Returns:
            归一化后的分数
        """
        if metric_config.mode == "minimize":
            # 对于需要最小化的指标，值越小分数越高
            cap = metric_config.soft_cap or 1.0
            ratio = min(raw_value / cap, 1.0)
            normalized = 1.0 - ratio
        else:
            # 对于需要最大化的指标，值越大分数越高
            floor = metric_config.soft_floor or 0.0
            clipped = max(raw_value, floor)
            # 假设最大值为1.0（对于R2等指标）
            normalized = min(clipped, 1.0)
        
        return normalized * metric_config.weight
    
    def compute_composite_score(self, metrics_values: Dict[str, float]) -> float:
        """
        计算组合分数
        
        Args:
            metrics_values: 指标值字典 {metric_name: value}
            
        Returns:
            组合分数
        """
        total_score = 0.0
        for metric_name, metric_config in self.metrics.items():
            if metric_name in metrics_values:
                raw_value = metrics_values[metric_name]
                total_score += self.normalize_score(raw_value, metric_config)
        
        return round(total_score, 6)
    
    def evaluate_improvement(
        self, 
        before: Dict[str, float], 
        after: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        评估改进情况
        
        Args:
            before: 优化前的指标值
            after: 优化后的指标值
            
        Returns:
            改进评估结果
        """
        before_score = self.compute_composite_score(before)
        after_score = self.compute_composite_score(after)
        
        improvement = after_score - before_score
        is_improved = improvement > 0
        
        # 计算各指标的改进
        metric_changes = {}
        for metric_name in self.metrics:
            if metric_name in before and metric_name in after:
                change = after[metric_name] - before[metric_name]
                metric_config = self.metrics[metric_name]
                # 对于minimize指标，减少是改进
                if metric_config.mode == "minimize":
                    is_metric_improved = change < 0
                else:
                    is_metric_improved = change > 0
                
                metric_changes[metric_name] = {
                    "before": before[metric_name],
                    "after": after[metric_name],
                    "change": change,
                    "improved": is_metric_improved
                }
        
        return {
            "composite_score_before": before_score,
            "composite_score_after": after_score,
            "improvement": improvement,
            "is_improved": is_improved,
            "metric_changes": metric_changes
        }
    
    def get_target_metrics(self) -> List[str]:
        """获取目标指标列表"""
        return list(self.metrics.keys())
    
    def get_metric_info(self, metric_name: str) -> Optional[Dict]:
        """获取指标信息"""
        if metric_name in self.metrics:
            cfg = self.metrics[metric_name]
            return {
                "name": cfg.name,
                "weight": cfg.weight,
                "mode": cfg.mode,
                "soft_cap": cfg.soft_cap,
                "soft_floor": cfg.soft_floor
            }
        return None
    
    def create_baseline_snapshot(
        self, 
        metrics_values: Dict[str, float],
        output_path: str
    ) -> Dict:
        """
        创建基线快照
        
        Args:
            metrics_values: 指标值
            output_path: 输出路径
            
        Returns:
            快照数据
        """
        composite_score = self.compute_composite_score(metrics_values)
        
        snapshot = {
            "composite_score": composite_score,
            "metrics": metrics_values,
            "config": {
                name: cfg.to_dict() 
                for name, cfg in self.metrics.items()
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        return snapshot


# 便捷函数
def create_default_definer() -> MetricsDefiner:
    """创建默认指标定义器"""
    return MetricsDefiner()
