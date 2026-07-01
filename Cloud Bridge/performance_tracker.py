# performance_tracker.py - Comprehensive tracking for time, tokens, latency, and cost
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from datetime import datetime

@dataclass
class StepMetrics:
    """Metrics for a single step"""
    step_name: str
    start_time: float
    end_time: float
    duration: float
    model: Optional[str] = None 
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0  
    
    def to_dict(self):
        result = {
            "step_name": self.step_name,
            "duration_seconds": round(self.duration, 3),
        }
        
        # Only include API-related fields if this was an API call
        if self.model:
            result.update({
                "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self.cost, 6),
                "latency_seconds": round(self.latency, 3)
            })
        else:
            # Non-API step - just include minimal info
            result["model"] = None
            result["total_tokens"] = 0
            result["cost_usd"] = 0.0
        
        return result

@dataclass
class PipelineMetrics:
    """Aggregated metrics for entire pipeline"""
    test_case_name: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[StepMetrics] = field(default_factory=list)
    
    @property
    def total_duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def total_tokens(self) -> int:
        return sum(step.total_tokens for step in self.steps)
    
    @property
    def total_cost(self) -> float:
        return sum(step.cost for step in self.steps)
    
    @property
    def total_prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)
    
    @property
    def total_completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)
    
    def to_dict(self):
        return {
            "test_case_name": self.test_case_name,
            "total_duration_seconds": round(self.total_duration, 3),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "steps": [step.to_dict() for step in self.steps],
            "step_count": len(self.steps)
        }

class PerformanceTracker:
    """Tracks performance metrics for API calls"""
    
    # Pricing per 1M tokens (updated for 2026)
    PRICING = {
        "gpt-5.1": {
            "prompt": 1.00,
            "completion": 4.00
        },
        "gpt-4o": {
            "prompt": 2.50,
            "completion": 10.00
        },
        "gpt-4o-mini": {
            "prompt": 0.075,
            "completion": 0.300
        },
        "gpt-4-turbo": {
            "prompt": 10.00,
            "completion": 30.00
        },
        "gpt-3.5-turbo": {
            "prompt": 0.50,
            "completion": 1.50
        },
        "o1": {
            "prompt": 15.00,
            "completion": 60.00
        },
        "o1-mini": {
            "prompt": 3.00,
            "completion": 12.00
        },
        "o3-mini-low": {
            "prompt": 1.10,
            "completion": 1.10
        },
        "o3-mini-medium": {
            "prompt": 2.20,
            "completion": 2.20
        },
        "o3-mini-high": {
            "prompt": 4.40,
            "completion": 4.40
        }
    }
    
    def __init__(self):
        self.current_pipeline: Optional[PipelineMetrics] = None
        self.all_pipelines: List[PipelineMetrics] = []
    
    def start_pipeline(self, test_case_name: str):
        """Start tracking a new pipeline"""
        self.current_pipeline = PipelineMetrics(
            test_case_name=test_case_name,
            start_time=time.time()
        )
    
    def end_pipeline(self):
        """End current pipeline tracking"""
        if self.current_pipeline:
            self.current_pipeline.end_time = time.time()
            self.all_pipelines.append(self.current_pipeline)
            result = self.current_pipeline
            self.current_pipeline = None
            return result
        return None
    
    def track_api_call(self, step_name: str, model: str, response, start_time: float = None) -> StepMetrics:
        """Track a single API call and return metrics"""
        end_time = time.time()
        
        # Use provided start_time or estimate (for backwards compatibility)
        if start_time is None:
            start_time = end_time - 0.5  
        
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        duration = end_time - start_time
        latency = min(0.1, duration)  # Latency is at most the duration
        
        step_metrics = StepMetrics(
            step_name=step_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency=latency
        )
        
        if self.current_pipeline:
            self.current_pipeline.steps.append(step_metrics)
        
        return step_metrics
    
    def track_non_api_step(self, step_name: str) -> StepMetrics:
        """Track a non-API step (pure processing/logic)"""
        step_metrics = StepMetrics(
            step_name=step_name,
            start_time=0,  
            end_time=0,
            duration=0,    
            model=None,    
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.0,
            latency=0.0
        )
        
        if self.current_pipeline:
            self.current_pipeline.steps.append(step_metrics)
        
        return step_metrics
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage"""
        if model not in self.PRICING:
            model = "gpt-4o"
        
        pricing = self.PRICING[model]
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def get_summary(self) -> Dict:
        """Get summary of all tracked pipelines"""
        if not self.all_pipelines:
            return {
                "total_test_cases": 0,
                "total_duration_seconds": 0,
                "total_tokens": 0,
                "total_cost_usd": 0,
                "average_duration_per_case": 0,
                "average_tokens_per_case": 0,
                "average_cost_per_case": 0,
                "pipelines": []
            }
        
        total_duration = sum(p.total_duration for p in self.all_pipelines)
        total_tokens = sum(p.total_tokens for p in self.all_pipelines)
        total_cost = sum(p.total_cost for p in self.all_pipelines)
        
        return {
            "total_test_cases": len(self.all_pipelines),
            "total_duration_seconds": round(total_duration, 3),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "average_duration_per_case": round(total_duration / len(self.all_pipelines), 3),
            "average_tokens_per_case": round(total_tokens / len(self.all_pipelines), 2),
            "average_cost_per_case": round(total_cost / len(self.all_pipelines), 6),
            "pipelines": [p.to_dict() for p in self.all_pipelines]
        }
    
    def save_report(self, filename: str = "performance_report.json"):
        """Save detailed performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename
    
    def print_summary(self):
        """Print a formatted summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("📊 PERFORMANCE SUMMARY")
        print("="*80)
        
        if summary['total_test_cases'] == 0:
            print("⚠️  No pipelines tracked yet.")
            print("="*80 + "\n")
            return
        
        print(f"\n🎯 Overall Metrics:")
        print(f"   Total Test Cases:        {summary['total_test_cases']}")
        print(f"   Total Duration:          {summary['total_duration_seconds']:.2f}s")
        print(f"   Total Tokens:            {summary['total_tokens']:,}")
        print(f"   Total Cost:              ${summary['total_cost_usd']:.6f}")
        
        print(f"\n📈 Averages per Test Case:")
        print(f"   Duration:                {summary['average_duration_per_case']:.2f}s")
        print(f"   Tokens:                  {summary['average_tokens_per_case']:,.0f}")
        print(f"   Cost:                    ${summary['average_cost_per_case']:.6f}")
        
        print(f"\n📋 Step-by-Step Breakdown:")
        for i, pipeline in enumerate(self.all_pipelines, 1):
            print(f"\n   Test Case {i}: {pipeline.test_case_name}")
            print(f"   {'='*76}")
            for step in pipeline.steps:
                if step.model:  # API call
                    print(f"   {step.step_name:40s} | "
                          f"{step.duration:6.2f}s | "
                          f"{step.total_tokens:6,} tokens | "
                          f"${step.cost:.6f}")
                else:  # Non-API step
                    print(f"   {step.step_name:40s} | "
                          f"{step.duration:6.2f}s | "
                          f"{'N/A':>6s} tokens | "
                          f"{'N/A':>10s}")
            print(f"   {'-'*76}")
            print(f"   {'TOTAL':40s} | "
                  f"{pipeline.total_duration:6.2f}s | "
                  f"{pipeline.total_tokens:6,} tokens | "
                  f"${pipeline.total_cost:.6f}")
        
        print("\n" + "="*80 + "\n")


class StepTimer:
    """Context manager for timing individual steps"""
    def __init__(self, tracker: PerformanceTracker, step_name: str, is_api_call: bool = False):
        self.tracker = tracker
        self.step_name = step_name
        self.is_api_call = is_api_call
        self.start_time = None
        self.step_metrics = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting: {self.step_name}")
        
        # For non-API steps, create the metrics now
        if not self.is_api_call:
            self.step_metrics = self.tracker.track_non_api_step(self.step_name)
        
        return self
    
    def set_metrics(self, metrics: StepMetrics):
        """Store reference to the metrics object for this step"""
        self.step_metrics = metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        print(f"✅ Completed: {self.step_name} ({duration:.2f}s)")
        
        if self.step_metrics:
            # Only overwrite if duration has not already been set by track_api_call
            if self.step_metrics.duration == 0:
                self.step_metrics.start_time = self.start_time
                self.step_metrics.duration = duration
        elif self.tracker.current_pipeline and self.tracker.current_pipeline.steps:
            for step in reversed(self.tracker.current_pipeline.steps):
                if step.step_name == self.step_name and step.duration == 0:
                    step.start_time = self.start_time
                    step.duration = duration
                    break