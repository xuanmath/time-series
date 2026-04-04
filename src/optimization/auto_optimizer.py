"""
自动优化模块
Auto Optimizer Module

自动分析代码/模型并生成优化策略
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import yaml


@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    strategy_name: str
    changes_made: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class OptimizationStrategy:
    """优化策略"""
    name: str
    description: str
    priority: int  # 优先级，数字越小越优先
    apply_func: Callable  # 应用优化的函数
    condition_func: Optional[Callable] = None  # 判断是否需要应用的条件


class AutoOptimizer:
    """
    自动优化器
    
    负责自动分析代码/模型并应用优化策略
    """
    
    def __init__(
        self,
        project_root: str = ".",
        config_path: Optional[str] = None
    ):
        """
        初始化自动优化器
        
        Args:
            project_root: 项目根目录
            config_path: 配置文件路径
        """
        self.project_root = Path(project_root)
        self.config_path = config_path or "configs/optimization.yaml"
        self.strategies: List[OptimizationStrategy] = []
        self.optimization_history: List[OptimizationResult] = []
        self.backup_dir = self.project_root / ".optimization_backup"
        
        self._load_config()
        self._register_default_strategies()
    
    def _load_config(self):
        """加载配置"""
        config_file = self.project_root / self.config_path
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {
                "max_iterations": 10,
                "backup_enabled": True,
                "auto_rollback": True,
                "strategies": {
                    "hyperparameter_tuning": True,
                    "feature_engineering": True,
                    "model_selection": True,
                    "code_refactoring": False
                }
            }
    
    def _register_default_strategies(self):
        """注册默认优化策略"""
        # 策略1: 超参数调优
        self.register_strategy(OptimizationStrategy(
            name="hyperparameter_tuning",
            description="自动调整模型超参数",
            priority=1,
            apply_func=self._optimize_hyperparameters,
            condition_func=self._should_tune_hyperparameters
        ))
        
        # 策略2: 特征工程优化
        self.register_strategy(OptimizationStrategy(
            name="feature_engineering",
            description="优化特征工程流程",
            priority=2,
            apply_func=self._optimize_features,
            condition_func=self._should_optimize_features
        ))
        
        # 策略3: 模型选择
        self.register_strategy(OptimizationStrategy(
            name="model_selection",
            description="尝试不同的模型架构",
            priority=3,
            apply_func=self._optimize_model_selection,
            condition_func=self._should_try_different_model
        ))
        
        # 策略4: 数据预处理优化
        self.register_strategy(OptimizationStrategy(
            name="data_preprocessing",
            description="优化数据预处理流程",
            priority=4,
            apply_func=self._optimize_preprocessing,
            condition_func=self._should_optimize_preprocessing
        ))
    
    def register_strategy(self, strategy: OptimizationStrategy):
        """注册优化策略"""
        self.strategies.append(strategy)
        # 按优先级排序
        self.strategies.sort(key=lambda s: s.priority)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """
        创建备份
        
        Args:
            backup_name: 备份名称
            
        Returns:
            备份路径
        """
        if not self.config.get("backup_enabled", True):
            return None
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        # 备份关键文件
        key_dirs = ["src", "configs", "scripts"]
        key_files = ["pyproject.toml", "requirements.txt"]
        
        for dir_name in key_dirs:
            src_dir = self.project_root / dir_name
            if src_dir.exists():
                dst_dir = backup_path / dir_name
                shutil.copytree(src_dir, dst_dir)
        
        for file_name in key_files:
            src_file = self.project_root / file_name
            if src_file.exists():
                dst_file = backup_path / file_name
                shutil.copy2(src_file, dst_file)
        
        return backup_path
    
    def restore_backup(self, backup_path: Path) -> bool:
        """
        从备份恢复
        
        Args:
            backup_path: 备份路径
            
        Returns:
            是否成功
        """
        if not backup_path.exists():
            return False
        
        try:
            # 恢复文件
            for item in backup_path.iterdir():
                dst = self.project_root / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
            return True
        except Exception as e:
            print(f"恢复备份失败: {e}")
            return False
    
    def _should_tune_hyperparameters(self, context: Dict) -> bool:
        """判断是否需要调参"""
        metrics = context.get("current_metrics", {})
        r2 = metrics.get("r2", 1.0)
        rmse = metrics.get("rmse", 0.0)
        return r2 < 0.9 or rmse > 0.1
    
    def _optimize_hyperparameters(self, context: Dict) -> OptimizationResult:
        """执行超参数优化"""
        changes = []
        
        config_file = self.project_root / "configs" / "config.yaml"
        if not config_file.exists():
            return OptimizationResult(
                success=False,
                strategy_name="hyperparameter_tuning",
                changes_made=[],
                metrics_before=context.get("current_metrics", {}),
                error_message="配置文件不存在"
            )
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        metrics = context.get("current_metrics", {})
        model_type = config.get("model", {}).get("name", "lstm")
        
        if model_type == "lstm":
            lstm_config = config["model"]["lstm"]
            
            if metrics.get("r2", 0) < 0.85:
                old_dropout = lstm_config.get("dropout", 0.1)
                new_dropout = min(old_dropout + 0.1, 0.5)
                lstm_config["dropout"] = new_dropout
                changes.append(f"dropout: {old_dropout} -> {new_dropout}")
            
            old_lr = lstm_config.get("learning_rate", 0.001)
            if metrics.get("rmse", 1.0) > 0.2:
                new_lr = old_lr * 0.5
                lstm_config["learning_rate"] = new_lr
                changes.append(f"learning_rate: {old_lr} -> {new_lr}")
            
            old_epochs = lstm_config.get("epochs", 100)
            if metrics.get("r2", 0) < 0.8:
                new_epochs = min(old_epochs + 50, 500)
                lstm_config["epochs"] = new_epochs
                changes.append(f"epochs: {old_epochs} -> {new_epochs}")
        
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return OptimizationResult(
            success=True,
            strategy_name="hyperparameter_tuning",
            changes_made=changes,
            metrics_before=context.get("current_metrics", {})
        )
    
    def _should_optimize_features(self, context: Dict) -> bool:
        """判断是否需要优化特征"""
        metrics = context.get("current_metrics", {})
        return metrics.get("r2", 1.0) < 0.88
    
    def _optimize_features(self, context: Dict) -> OptimizationResult:
        """执行特征工程优化"""
        changes = []
        
        config_file = self.project_root / "configs" / "config.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        features_config = config.get("features", {})
        
        current_lags = features_config.get("lags", [1, 7, 14, 28])
        new_lags = list(set(current_lags + [3, 5, 10, 21]))
        features_config["lags"] = sorted(new_lags)
        changes.append(f"lags updated to {sorted(new_lags)}")
        
        current_windows = features_config.get("rolling_windows", [7, 14, 28])
        new_windows = list(set(current_windows + [3, 5, 10]))
        features_config["rolling_windows"] = sorted(new_windows)
        changes.append(f"rolling_windows updated to {sorted(new_windows)}")
        
        config["features"] = features_config
        
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return OptimizationResult(
            success=True,
            strategy_name="feature_engineering",
            changes_made=changes,
            metrics_before=context.get("current_metrics", {})
        )
    
    def _should_try_different_model(self, context: Dict) -> bool:
        """判断是否需要尝试不同模型"""
        metrics = context.get("current_metrics", {})
        iteration = context.get("iteration", 0)
        return metrics.get("r2", 1.0) < 0.8 and iteration >= 3
    
    def _optimize_model_selection(self, context: Dict) -> OptimizationResult:
        """执行模型选择优化"""
        changes = []
        
        config_file = self.project_root / "configs" / "config.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        model_sequence = ["lstm", "gru", "transformer", "prophet"]
        current_model = config.get("model", {}).get("name", "lstm")
        
        try:
            current_idx = model_sequence.index(current_model)
            next_idx = (current_idx + 1) % len(model_sequence)
            next_model = model_sequence[next_idx]
        except ValueError:
            next_model = "lstm"
        
        config["model"]["name"] = next_model
        changes.append(f"model: {current_model} -> {next_model}")
        
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return OptimizationResult(
            success=True,
            strategy_name="model_selection",
            changes_made=changes,
            metrics_before=context.get("current_metrics", {})
        )
    
    def _should_optimize_preprocessing(self, context: Dict) -> bool:
        """判断是否需要优化预处理"""
        metrics = context.get("current_metrics", {})
        return metrics.get("mae", 0) > 0.1
    
    def _optimize_preprocessing(self, context: Dict) -> OptimizationResult:
        """执行预处理优化"""
        changes = []
        
        config_file = self.project_root / "configs" / "config.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        data_config = config.get("data", {})
        old_test_size = data_config.get("test_size", 0.2)
        
        if old_test_size > 0.15:
            new_test_size = old_test_size - 0.05
            data_config["test_size"] = new_test_size
            changes.append(f"test_size: {old_test_size} -> {new_test_size}")
        
        config["data"] = data_config
        
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return OptimizationResult(
            success=True,
            strategy_name="data_preprocessing",
            changes_made=changes,
            metrics_before=context.get("current_metrics", {})
        )
    
    def run_optimization(
        self,
        current_metrics: Dict[str, float],
        iteration: int = 0
    ) -> List[OptimizationResult]:
        """
        运行优化流程
        
        Args:
            current_metrics: 当前指标值
            iteration: 当前迭代次数
            
        Returns:
            优化结果列表
        """
        results = []
        context = {
            "current_metrics": current_metrics,
            "iteration": iteration
        }
        
        backup_path = self.create_backup(f"pre_optimization_iter{iteration}")
        
        for strategy in self.strategies:
            if not self.config.get("strategies", {}).get(strategy.name, True):
                continue
            
            if strategy.condition_func and not strategy.condition_func(context):
                continue
            
            result = strategy.apply_func(context)
            results.append(result)
            self.optimization_history.append(result)
            
            if result.success:
                print(f"[OK] Strategy '{strategy.name}' executed successfully")
                for change in result.changes_made:
                    print(f"  - {change}")
            else:
                print(f"[FAIL] Strategy '{strategy.name}' failed: {result.error_message}")
        
        return results
    
    def get_optimization_summary(self) -> Dict:
        """获取优化历史摘要"""
        total = len(self.optimization_history)
        successful = sum(1 for r in self.optimization_history if r.success)
        
        return {
            "total_optimizations": total,
            "successful": successful,
            "failed": total - successful,
            "strategies_applied": list(set(r.strategy_name for r in self.optimization_history))
        }
    
    def save_history(self, output_path: str):
        """保存优化历史"""
        history_data = {
            "summary": self.get_optimization_summary(),
            "history": [
                {
                    "strategy": r.strategy_name,
                    "success": r.success,
                    "changes": r.changes