# model_optimizer.py - PRODUCTION: OpenAI-only intelligent model selection
from typing import Dict, Literal, Optional, List
from openai import OpenAI
import re
from dataclasses import dataclass
from enum import Enum

class OptimizationStrategy(Enum):
    """Optimization strategies for model selection"""
    COST_FOCUSED = "cost"      # Maximize cost savings (use cheaper models when possible)
    SPEED_FOCUSED = "speed"    # Maximize speed (use faster models)
    BALANCED = "balanced"      # Balance cost, speed, and quality
    QUALITY_FOCUSED = "quality" # Always use best models

@dataclass
class ModelSpec:
    """Specification for a model including costs and capabilities"""
    name: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    relative_speed: float  # 1.0 = baseline, <1.0 = faster, >1.0 = slower
    capability_tier: int   # 1=highest, 2=high, 3=medium, 4=basic
    max_tokens: int
    context_window: int
    recommended_for: List[str]
    notes: str = ""

class ModelOptimizer:
    """
    Intelligently selects optimal OpenAI models based on:
    - Task complexity analysis
    - Cost optimization strategies  
    - Speed requirements
    - Quality needs
    - Context length requirements
    """
    
    # ============================================================
    # OPENAI MODEL CATALOG - All available models
    # ============================================================
    MODELS = {
        # -------------------- TIER 1: HIGHEST CAPABILITY --------------------
        "gpt-5.1": ModelSpec(
        name="gpt-5.1",
        cost_per_1m_input=1.50,       # updated pricing tier
        cost_per_1m_output=6.00,      # updated pricing tier
        relative_speed=1.6,           # faster than 4o
        capability_tier=1,            # top-tier model
        max_tokens=32768,             # higher per-request limit
        context_window=200000,        # expanded context
        recommended_for=[
            "evaluation",
            "comparison",
            "strategic_analysis",
            "complex_reasoning",
            "brand_strategy",
            "nuanced_analysis",
            "creative_writing",
            "long_context_tasks",
            "research_assistance",
            "higher-precision_judgment",
        ],
        notes="Flagship 5-series model. Best for long-context reasoning, precision evaluation, and advanced generation."
    ),
    
        "gpt-4o": ModelSpec(
            name="gpt-4o",
            cost_per_1m_input=2.50,
            cost_per_1m_output=10.00,
            relative_speed=1.0,
            capability_tier=1,
            max_tokens=16384,
            context_window=128000,
            recommended_for=["evaluation", "comparison", "strategic_analysis", "complex_reasoning", 
                           "brand_strategy", "nuanced_analysis", "creative_writing"],
            notes="Most capable model - best for complex evaluation and strategic tasks"
        ),
        
        "gpt-4-turbo": ModelSpec(
            name="gpt-4-turbo",
            cost_per_1m_input=10.00,
            cost_per_1m_output=30.00,
            relative_speed=1.3,
            capability_tier=1,
            max_tokens=4096,
            context_window=128000,
            recommended_for=["legacy_evaluation", "complex_analysis"],
            notes="Previous gen GPT-4 - expensive, use gpt-4o instead"
        ),
        
        "gpt-4": ModelSpec(
            name="gpt-4",
            cost_per_1m_input=30.00,
            cost_per_1m_output=60.00,
            relative_speed=1.5,
            capability_tier=1,
            max_tokens=8192,
            context_window=8192,
            recommended_for=["legacy_tasks"],
            notes="Original GPT-4 - very expensive, prefer gpt-4o"
        ),
        
        "o1-preview": ModelSpec(
            name="o1-preview",
            cost_per_1m_input=15.00,
            cost_per_1m_output=60.00,
            relative_speed=2.0,  # Slower due to reasoning
            capability_tier=1,
            max_tokens=32768,
            context_window=128000,
            recommended_for=["complex_reasoning", "mathematical_problems", "strategic_planning", 
                           "multi_step_logic", "research_analysis"],
            notes="Advanced reasoning model - best for complex logical tasks requiring deep thought"
        ),
        
        "o1-mini": ModelSpec(
            name="o1-mini",
            cost_per_1m_input=3.00,
            cost_per_1m_output=12.00,
            relative_speed=1.5,
            capability_tier=1,
            max_tokens=65536,
            context_window=128000,
            recommended_for=["moderate_reasoning", "problem_solving", "logical_analysis"],
            notes="Faster reasoning model for moderate complexity reasoning tasks"
        ),
        
        # -------------------- TIER 2: HIGH CAPABILITY (BALANCED) --------------------
        
        "gpt-4o-mini": ModelSpec(
            name="gpt-4o-mini",
            cost_per_1m_input=0.150,
            cost_per_1m_output=0.600,
            relative_speed=0.7,
            capability_tier=2,
            max_tokens=16384,
            context_window=128000,
            recommended_for=["context_analysis", "tone_detection", "classification", 
                           "response_generation", "simple_evaluation", "content_extraction",
                           "emotional_analysis", "moderate_complexity"],
            notes="★ BEST BALANCE - Fast, cheap, very capable. Use for most tasks!"
        ),
        
        "gpt-4o-2024-08-06": ModelSpec(
            name="gpt-4o-2024-08-06",
            cost_per_1m_input=2.50,
            cost_per_1m_output=10.00,
            relative_speed=1.0,
            capability_tier=1,
            max_tokens=16384,
            context_window=128000,
            recommended_for=["structured_outputs", "function_calling"],
            notes="Specific version of gpt-4o with enhanced structured outputs"
        ),
        
        # -------------------- TIER 3: MEDIUM CAPABILITY (FAST & CHEAP) --------------------
        
        "gpt-3.5-turbo": ModelSpec(
            name="gpt-3.5-turbo",
            cost_per_1m_input=0.50,
            cost_per_1m_output=1.50,
            relative_speed=0.5,
            capability_tier=3,
            max_tokens=4096,
            context_window=16385,
            recommended_for=["simple_classification", "basic_extraction", "quick_validation", 
                           "scope_checking", "simple_responses", "fast_filtering"],
            notes="Fast and cheap - good for simple, straightforward tasks"
        ),
        
        "gpt-3.5-turbo-16k": ModelSpec(
            name="gpt-3.5-turbo-16k",
            cost_per_1m_input=3.00,
            cost_per_1m_output=4.00,
            relative_speed=0.6,
            capability_tier=3,
            max_tokens=4096,
            context_window=16385,
            recommended_for=["long_context_simple_tasks", "document_processing"],
            notes="GPT-3.5 with larger context - use regular 3.5 unless you need 16k context"
        ),
        
        "gpt-3.5-turbo-0125": ModelSpec(
            name="gpt-3.5-turbo-0125",
            cost_per_1m_input=0.50,
            cost_per_1m_output=1.50,
            relative_speed=0.5,
            capability_tier=3,
            max_tokens=4096,
            context_window=16385,
            recommended_for=["simple_tasks", "json_outputs"],
            notes="Latest GPT-3.5 with better instruction following"
        ),
        
        # -------------------- TIER 4: BASIC (LEGACY/SPECIALIZED) --------------------
        
        "gpt-3.5-turbo-instruct": ModelSpec(
            name="gpt-3.5-turbo-instruct",
            cost_per_1m_input=1.50,
            cost_per_1m_output=2.00,
            relative_speed=0.5,
            capability_tier=4,
            max_tokens=4096,
            context_window=4096,
            recommended_for=["completions", "legacy_tasks"],
            notes="Completion model (not chat) - rarely needed"
        ),
        
        "davinci-002": ModelSpec(
            name="davinci-002",
            cost_per_1m_input=2.00,
            cost_per_1m_output=2.00,
            relative_speed=0.8,
            capability_tier=4,
            max_tokens=16384,
            context_window=16384,
            recommended_for=["legacy_completions", "fine_tuning"],
            notes="Legacy base model - prefer chat models"
        ),
        
        "babbage-002": ModelSpec(
            name="babbage-002",
            cost_per_1m_input=0.40,
            cost_per_1m_output=0.40,
            relative_speed=0.3,
            capability_tier=4,
            max_tokens=16384,
            context_window=16384,
            recommended_for=["ultra_simple_tasks", "fine_tuning_base"],
            notes="Very basic model - only for extremely simple tasks or fine-tuning"
        ),
    }
    
    # ============================================================
    # COMPLEXITY ANALYSIS - Determines minimum required tier
    # ============================================================
    COMPLEXITY_INDICATORS = {
        "critical": {  # Requires Tier 1 models
            "keywords": [
                "evaluate", "compare deeply", "strategic decision", "nuanced analysis",
                "comprehensive evaluation", "brand alignment", "tone strategy", 
                "deep reasoning", "complex logic", "multi-step analysis", "synthesize",
                "assess quality", "comparative analysis", "strategic thinking"
            ],
            "min_tier": 1
        },
        "high": {  # Tier 1-2 acceptable
            "keywords": [
                "analyze", "determine strategy", "emotional intelligence",
                "contextual understanding", "multi-factor", "generate response",
                "tone adaptation", "creative", "understand nuance", "assess"
            ],
            "min_tier": 2
        },
        "medium": {  # Tier 2-3 acceptable
            "keywords": [
                "identify", "extract", "categorize", "understand context",
                "recognize pattern", "classify", "summarize", "detect",
                "simple analysis", "content generation"
            ],
            "min_tier": 2
        },
        "low": {  # Any tier acceptable (will choose cheapest/fastest)
            "keywords": [
                "check", "validate", "simple", "yes/no", "binary decision",
                "straightforward", "basic", "quick", "filter", "scope",
                "verify", "confirm"
            ],
            "min_tier": 3
        }
    }
    
    def __init__(
        self, 
        client: OpenAI, 
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        cost_threshold: Optional[float] = None,
        speed_requirement: Optional[float] = None
    ):
        """
        Initialize ModelOptimizer
        
        Args:
            client: OpenAI client instance
            strategy: Optimization strategy to use
            cost_threshold: Maximum cost per request (optional)
            speed_requirement: Maximum acceptable latency multiplier (optional)
        """
        self.client = client
        self.strategy = strategy
        self.cost_threshold = cost_threshold
        self.speed_requirement = speed_requirement
        self.selection_history: List[Dict] = []
        self.selection_cache: Dict[str, str] = {}
        
    def select_model(
        self, 
        step_name: str, 
        prompt: str = "", 
        force_model: Optional[str] = None,
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: Optional[int] = None,
        require_long_context: bool = False
    ) -> str:
        """
        Select optimal model based on task requirements
        
        Args:
            step_name: Name of the processing step (helps determine complexity)
            prompt: The prompt text (analyzed for complexity)
            force_model: Force a specific model (bypasses optimization)
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens
            require_long_context: If True, requires models with 100k+ context
        
        Returns:
            Selected model name
        """
        # Force model if specified
        if force_model:
            if force_model in self.MODELS:
                self._log_selection(step_name, force_model, "forced", {})
                return force_model
            else:
                print(f"⚠️  Unknown model '{force_model}', using default selection")
        # Force model if specified

        eval_keywords = ["evaluation", "evaluate", "tone strategy", "tone_strategy", "tone-strategy"]
        step_lower = step_name.lower()

        if any(k in step_lower for k in eval_keywords):
            self._log_selection(step_name, "gpt-5.1", "forced-eval", {})
            return "gpt-5.1"

        
        # Check cache
        cache_key = self._get_cache_key(step_name, prompt, require_long_context)
        if cache_key in self.selection_cache:
            cached_model = self.selection_cache[cache_key]
            self._log_selection(step_name, cached_model, "cached", {})
            return cached_model
        
        # Analyze task complexity
        complexity = self._analyze_complexity(step_name, prompt)
        required_tier = self._get_required_tier(complexity)
        
        # Estimate tokens if not provided
        if estimated_input_tokens is None:
            estimated_input_tokens = int(len(prompt.split()) * 1.3)
        if estimated_output_tokens is None:
            estimated_output_tokens = 500
        
        # Get eligible models
        eligible_models = self._get_eligible_models(
            required_tier, 
            estimated_input_tokens,
            require_long_context
        )
        
        if not eligible_models:
            # Fallback to most capable model
            selected_model = "gpt-4o"
        else:
            # Apply optimization strategy
            selected_model = self._apply_strategy(
                eligible_models,
                estimated_input_tokens,
                estimated_output_tokens,
                complexity
            )
        
        # Cache and log
        self.selection_cache[cache_key] = selected_model
        self._log_selection(step_name, selected_model, complexity, {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "strategy": self.strategy.value,
            "require_long_context": require_long_context
        })
        
        return selected_model
    
    def _analyze_complexity(self, step_name: str, prompt: str) -> str:
        """Analyze task complexity from step name and prompt"""
        text = f"{step_name} {prompt}".lower()
        
        # Check indicators for each complexity level
        for level in ["critical", "high", "medium", "low"]:
            indicators = self.COMPLEXITY_INDICATORS[level]["keywords"]
            matches = sum(1 for keyword in indicators if keyword in text)
            
            # Critical: match any 1 keyword
            if level == "critical" and matches >= 1:
                return "critical"
            
            # Others: match at least 2 keywords
            if matches >= 2:
                return level
        
        # Fallback: pattern matching on step name
        step_lower = step_name.lower()
        if any(k in step_lower for k in ["evaluat", "compar", "strateg", "reason"]):
            return "critical"
        if any(k in step_lower for k in ["analyz", "determin", "assess", "tone", "context"]):
            return "high"
        if any(k in step_lower for k in ["extract", "identify", "generat", "response"]):
            return "medium"
        if any(k in step_lower for k in ["validat", "check", "scope", "verify"]):
            return "low"
        
        return "medium"  # Safe default
    
    def _get_required_tier(self, complexity: str) -> int:
        """Get minimum required model tier for complexity level"""
        return self.COMPLEXITY_INDICATORS[complexity]["min_tier"]
    
    def _get_eligible_models(
        self, 
        min_tier: int, 
        estimated_tokens: int,
        require_long_context: bool
    ) -> List[ModelSpec]:
        """Filter models that meet all requirements"""
        eligible = []
        
        for spec in self.MODELS.values():
            # Must meet capability tier requirement
            if spec.capability_tier > min_tier:
                continue
            
            # Must have sufficient context window
            if estimated_tokens > spec.context_window * 0.8:  # 80% safety margin
                continue
            
            # Must meet long context requirement if specified
            if require_long_context and spec.context_window < 100000:
                continue
            
            eligible.append(spec)
        
        return eligible
    
    def _apply_strategy(
        self,
        eligible_models: List[ModelSpec],
        input_tokens: int,
        output_tokens: int,
        complexity: str
    ) -> str:
        """Select best model from eligible models based on strategy"""
        
        if not eligible_models:
            return "gpt-4o"
        
        if len(eligible_models) == 1:
            return eligible_models[0].name
        
        # Score each model
        scored_models = []
        for model in eligible_models:
            score = self._calculate_model_score(
                model, input_tokens, output_tokens
            )
            scored_models.append((model.name, score, model))
        
        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[0][0]
    
    def _calculate_model_score(
        self,
        model: ModelSpec,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate model score based on optimization strategy"""
        
        # Calculate actual cost for this request
        cost = (
            (input_tokens / 1_000_000) * model.cost_per_1m_input +
            (output_tokens / 1_000_000) * model.cost_per_1m_output
        )
        
        # Normalize factors to 0-1 scale (higher is better)
        cost_score = 1 / (1 + cost * 10000)  # Lower cost = higher score
        speed_score = 1 / model.relative_speed  # Faster = higher score
        quality_score = 1 / model.capability_tier  # Lower tier number = higher quality
        
        # Apply strategy-specific weights
        if self.strategy == OptimizationStrategy.COST_FOCUSED:
            # Heavily favor cost savings
            return 0.70 * cost_score + 0.20 * speed_score + 0.10 * quality_score
        
        elif self.strategy == OptimizationStrategy.SPEED_FOCUSED:
            # Heavily favor speed
            return 0.15 * cost_score + 0.75 * speed_score + 0.10 * quality_score
        
        elif self.strategy == OptimizationStrategy.QUALITY_FOCUSED:
            # Heavily favor quality (will usually pick tier 1 models)
            return 0.10 * cost_score + 0.10 * speed_score + 0.80 * quality_score
        
        else:  # BALANCED (default)
            # Balanced consideration of all factors
            return 0.40 * cost_score + 0.30 * speed_score + 0.30 * quality_score
    
    def _get_cache_key(self, step_name: str, prompt: str, require_long: bool) -> str:
        """Generate cache key for model selection"""
        prompt_hash = hash(prompt[:100]) if prompt else 0
        return f"{step_name}:{len(prompt)}:{require_long}:{prompt_hash}"
    
    def _log_selection(self, step_name: str, model: str, reason: str, metadata: Dict):
        """Log model selection decision for analysis"""
        self.selection_history.append({
            "step_name": step_name,
            "selected_model": model,
            "reason": reason,
            "metadata": metadata
        })
    
    def get_model_info(self, model_name: str) -> Optional[ModelSpec]:
        """Get detailed information about a specific model"""
        return self.MODELS.get(model_name)
    
    def list_models_by_tier(self, tier: Optional[int] = None) -> List[Dict]:
        """List models organized by capability tier"""
        models = []
        for spec in self.MODELS.values():
            if tier is not None and spec.capability_tier != tier:
                continue
            
            models.append({
                "name": spec.name,
                "tier": spec.capability_tier,
                "cost_input": f"${spec.cost_per_1m_input:.2f}/1M",
                "cost_output": f"${spec.cost_per_1m_output:.2f}/1M",
                "speed": f"{spec.relative_speed}x",
                "context": f"{spec.context_window:,}",
                "best_for": ", ".join(spec.recommended_for[:3])
            })
        
        return sorted(models, key=lambda x: (x["tier"], x["name"]))
    
    def print_model_catalog(self):
        """Print beautiful catalog of all available models"""
        print("\n" + "="*110)
        print("📚 OPENAI MODEL CATALOG - All Available Models")
        print("="*110)
        
        for tier in [1, 2, 3, 4]:
            models = [s for s in self.MODELS.values() if s.capability_tier == tier]
            if not models:
                continue
            
            tier_names = {1: "HIGHEST CAPABILITY", 2: "HIGH CAPABILITY (Balanced)", 
                         3: "MEDIUM (Fast & Cheap)", 4: "BASIC (Legacy/Specialized)"}
            
            print(f"\n{'🔥' if tier == 1 else '⚡' if tier == 2 else '💨' if tier == 3 else '📦'} "
                  f"TIER {tier}: {tier_names[tier]}")
            print("-" * 110)
            
            for spec in sorted(models, key=lambda x: x.name):
                star = " ★" if spec.name == "gpt-4o-mini" else "  "
                print(f"{star} {spec.name:30s} │ "
                      f"In: ${spec.cost_per_1m_input:6.2f}/1M │ "
                      f"Out: ${spec.cost_per_1m_output:6.2f}/1M │ "
                      f"Speed: {spec.relative_speed:4.1f}x │ "
                      f"Context: {spec.context_window:>7,}")
                
                if spec.notes:
                    print(f"   └─ {spec.notes}")
        
        print("\n" + "="*110)
        print("💡 TIP: For your thesis, use gpt-4o (evaluation) + gpt-4o-mini (most tasks) + gpt-3.5-turbo (simple)")
        print("="*110 + "\n")
    
    def get_savings_summary(self) -> Dict:
        """Calculate total savings compared to always using most expensive model"""
        if not self.selection_history:
            return {
                "total_requests": 0,
                "estimated_cost_saved_usd": 0,
                "cost_savings_percentage": 0
            }
        
        # Compare against always using gpt-4-turbo (most expensive common model)
        baseline_model = self.MODELS["gpt-4-turbo"]
        
        total_baseline_cost = 0
        total_actual_cost = 0
        
        for selection in self.selection_history:
            selected_model = self.MODELS.get(selection["selected_model"])
            if not selected_model:
                continue
            
            input_tokens = selection["metadata"].get("estimated_input_tokens", 1000)
            output_tokens = selection["metadata"].get("estimated_output_tokens", 500)
            
            baseline_cost = (
                (input_tokens / 1_000_000) * baseline_model.cost_per_1m_input +
                (output_tokens / 1_000_000) * baseline_model.cost_per_1m_output
            )
            
            actual_cost = (
                (input_tokens / 1_000_000) * selected_model.cost_per_1m_input +
                (output_tokens / 1_000_000) * selected_model.cost_per_1m_output
            )
            
            total_baseline_cost += baseline_cost
            total_actual_cost += actual_cost
        
        savings = total_baseline_cost - total_actual_cost
        savings_pct = (savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
        
        return {
            "total_requests": len(self.selection_history),
            "baseline_cost_usd": round(total_baseline_cost, 6),
            "actual_cost_usd": round(total_actual_cost, 6),
            "estimated_cost_saved_usd": round(savings, 6),
            "cost_savings_percentage": round(savings_pct, 1)
        }
    
    def print_selection_summary(self):
        """Print detailed summary of model selections and savings"""
        if not self.selection_history:
            print("\n⚠️  No model selections recorded yet.")
            return
        
        print("\n" + "="*110)
        print("📊 MODEL SELECTION SUMMARY")
        print("="*110)
        
        print(f"\n📋 Optimization Strategy: {self.strategy.value.upper()}")
        print(f"🔢 Total Selections: {len(self.selection_history)}")
        
        # Model usage breakdown
        model_counts = {}
        for selection in self.selection_history:
            model = selection["selected_model"]
            model_counts[model] = model_counts.get(model, 0) + 1
        
        print(f"\n📈 Model Usage Distribution:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(self.selection_history)) * 100
            spec = self.MODELS.get(model)
            tier = spec.capability_tier if spec else "?"
            print(f"   • {model:30s} {count:3d} times ({pct:5.1f}%) - Tier {tier}")
        
        # Complexity breakdown
        complexity_counts = {}
        for selection in self.selection_history:
            reason = selection["reason"]
            if reason in ["forced", "cached"]:
                continue
            complexity_counts[reason] = complexity_counts.get(reason, 0) + 1
        
        if complexity_counts:
            print(f"\n🎯 Task Complexity Distribution:")
            for complexity, count in sorted(complexity_counts.items()):
                print(f"   • {complexity.capitalize():15s} {count:3d} tasks")
        
        # Cost savings
        savings = self.get_savings_summary()
        print(f"\n💰 Cost Analysis (vs always using gpt-4-turbo):")
        print(f"   • Baseline Cost:     ${savings['baseline_cost_usd']:.6f}")
        print(f"   • Actual Cost:       ${savings['actual_cost_usd']:.6f}")
        print(f"   • Amount Saved:      ${savings['estimated_cost_saved_usd']:.6f}")
        print(f"   • Savings:           {savings['cost_savings_percentage']:.1f}%")
        
        print("\n" + "="*110 + "\n")
    
    def set_strategy(self, strategy: OptimizationStrategy):
        """Change optimization strategy (clears cache)"""
        self.strategy = strategy
        self.selection_cache.clear()
        print(f"✓ Strategy changed to: {strategy.value.upper()}")
    
    def reset_cache(self):
        """Clear selection cache (forces re-evaluation)"""
        self.selection_cache.clear()
        print("✓ Selection cache cleared")
    
    def reset_history(self):
        """Clear selection history"""
        self.selection_history.clear()
        print("✓ Selection history cleared")


# ============================================================
# GLOBAL OPTIMIZER INSTANCE
# ============================================================
optimizer = None

def initialize_optimizer(
    client: OpenAI,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
) -> ModelOptimizer:
    """
    Initialize the global model optimizer
    
    Args:
        client: OpenAI client instance
        strategy: Optimization strategy (COST_FOCUSED, SPEED_FOCUSED, BALANCED, QUALITY_FOCUSED)
    
    Returns:
        ModelOptimizer instance
    
    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> optimizer = initialize_optimizer(client, OptimizationStrategy.COST_FOCUSED)
    """
    global optimizer
    optimizer = ModelOptimizer(client, strategy=strategy)
    print(f"✓ Model optimizer initialized with strategy: {strategy.value.upper()}")
    return optimizer