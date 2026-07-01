"""
Tone Adaptation Engine - Adapts communication style to context and user needs
Ensures every response matches CloudBridge brand voice while fitting the moment
CloudBridge Technology Solutions Edition
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from openai import OpenAI

class ToneProfile(Enum):
    """Standard tone profiles for different situations"""
    PROFESSIONAL = "professional"
    HELPFUL = "helpful"
    EMPATHETIC = "empathetic"
    SOLUTION_FOCUSED = "solution_focused"
    CALM_REASSURING = "calm_reassuring"
    EDUCATIONAL = "educational"
    NEUTRAL = "neutral"

@dataclass
@dataclass
class ToneParameters:
    """
    Specific tone settings for response generation
    Each parameter ranges from 0.0 to 1.0
    """
    formality: float      # 0=casual, 1=very formal (CloudBridge: 0.4-0.7)
    warmth: float         # 0=neutral, 1=very warm
    energy: float         # 0=calm, 1=high energy
    directness: float     # 0=gentle, 1=very direct
    empathy: float        # 0=neutral, 1=highly empathetic
    reassurance: float    # 0=neutral, 1=highly reassuring
    authority: float      # 0=peer-level, 1=authoritative
    
    vocabulary_complexity: str  # "simple", "moderate", "technical"
    sentence_structure: str     # "simple", "moderate", "complex"
    explanation_depth: str      # "brief", "standard", "detailed"
    
    # CloudBridge brand requirements
    use_contractions: bool      # Moderate contractions for approachability
    avoid_hype: bool            # Never use hype language
    focus_on_outcomes: bool     # Focus on what user can achieve
    
    use_bold_text: bool = True
    use_numbered_lists: bool = True
    preferred_bullet_style: str = "-"  
    
    def to_dict(self) -> Dict:
        return {
            "formality": self.formality,
            "warmth": self.warmth,
            "energy": self.energy,
            "directness": self.directness,
            "empathy": self.empathy,
            "reassurance": self.reassurance,
            "authority": self.authority,
            "vocabulary_complexity": self.vocabulary_complexity,
            "sentence_structure": self.sentence_structure,
            "explanation_depth": self.explanation_depth,
            "use_contractions": self.use_contractions,
            "avoid_hype": self.avoid_hype,
            "focus_on_outcomes": self.focus_on_outcomes,
            "use_bold_text": self.use_bold_text,
            "use_numbered_lists": self.use_numbered_lists,
            "preferred_bullet_style": self.preferred_bullet_style
        }

class ToneAdaptationEngine:
    """
    Selects the right tone for each conversation moment
    
    What it does:
    - Maps context to appropriate tone settings
    - Keeps tone within CloudBridge brand boundaries
    - Ensures smooth tone transitions across conversation
    - Validates tone consistency with brand voice
    
    Why it matters:
    The same information lands differently depending on how it's delivered.
    This ensures we're always professional yet approachable, clear yet empathetic.
    """
    
    def __init__(self, brand_boundaries: Dict, client: OpenAI):
        """
        Args:
            brand_boundaries: CloudBridge tone boundaries from BrandConsistencyGuard
            client: OpenAI client for AI-powered tone adjustments
        """
        self.brand_boundaries = brand_boundaries
        self.client = client
        
        # Define CloudBridge tone profiles
        self.tone_profiles = self._build_tone_profiles()
        
        # Context-to-tone mapping rules
        self.mapping_rules = self._build_mapping_rules()
        
        # Track tone history for smooth transitions
        self.tone_history: List[ToneParameters] = []
    
    def _build_tone_profiles(self) -> Dict[ToneProfile, ToneParameters]:
        """
        Define CloudBridge tone profiles
        
        All profiles maintain CloudBridge DNA:
        - Professional but not cold (formality: 0.4-0.7)
        - Empowering without being patronizing
        - Clear and outcome-focused
        - Moderate contractions for approachability
        """
        return {
            ToneProfile.PROFESSIONAL: ToneParameters(
                formality=0.6,       # Professional but still approachable
                warmth=0.6,          # Friendly without being casual
                energy=0.4,          # Measured and steady
                directness=0.7,      # Clear and straightforward
                empathy=0.6,         # Understanding and respectful
                reassurance=0.5,     # Confident without overpromising
                authority=0.6,       # Knowledgeable but not condescending
                vocabulary_complexity="moderate",
                sentence_structure="moderate",
                explanation_depth="standard",
                use_contractions=True,    # Natural, approachable
                avoid_hype=True,          # No exaggeration
                focus_on_outcomes=True    # What user can achieve
            ),
            
            ToneProfile.HELPFUL: ToneParameters(
                formality=0.4,       # More approachable
                warmth=0.8,          # Warm and supportive
                energy=0.5,          # Engaged but not overwhelming
                directness=0.6,      # Clear guidance
                empathy=0.7,         # Understanding their needs
                reassurance=0.7,     # "We'll help you figure this out"
                authority=0.5,       # Peer-level helpfulness
                vocabulary_complexity="simple",
                sentence_structure="simple",
                explanation_depth="standard",
                use_contractions=True,
                avoid_hype=True,
                focus_on_outcomes=True
            ),
            
            ToneProfile.EMPATHETIC: ToneParameters(
                formality=0.5,       # Professional but warm
                warmth=0.9,          # Very warm and understanding
                energy=0.4,          # Calm and reassuring
                directness=0.5,      # Gentle but clear
                empathy=0.95,        # Highly empathetic
                reassurance=0.8,     # "We understand and we're here"
                authority=0.4,       # Less authoritative, more supportive
                vocabulary_complexity="moderate",
                sentence_structure="moderate",
                explanation_depth="detailed",
                use_contractions=True,
                avoid_hype=True,
                focus_on_outcomes=True
            ),
            
            ToneProfile.SOLUTION_FOCUSED: ToneParameters(
                formality=0.5,       # Professional and action-oriented
                warmth=0.6,          # Supportive but focused
                energy=0.6,          # Energized to help
                directness=0.8,      # Very direct, get to solutions
                empathy=0.6,         # Understanding but action-oriented
                reassurance=0.7,     # Confident in solutions
                authority=0.6,       # Knowledgeable guidance
                vocabulary_complexity="moderate",
                sentence_structure="simple",
                explanation_depth="brief",
                use_contractions=True,
                avoid_hype=True,
                focus_on_outcomes=True
            ),
            
            ToneProfile.CALM_REASSURING: ToneParameters(
                formality=0.5,       # Professional but reassuring
                warmth=0.8,          # Warm and calming
                energy=0.3,          # Very calm and steady
                directness=0.7,      # Clear about what's happening
                empathy=0.8,         # Understanding their concern
                reassurance=0.9,     # Highly reassuring
                authority=0.7,       # Authoritative confidence
                vocabulary_complexity="moderate",
                sentence_structure="moderate",
                explanation_depth="detailed",
                use_contractions=True,
                avoid_hype=True,
                focus_on_outcomes=True
            ),
            
            ToneProfile.EDUCATIONAL: ToneParameters(
                formality=0.5,       # Professional but accessible
                warmth=0.7,          # Warm and patient
                energy=0.4,          # Calm and patient
                directness=0.5,      # Clear but not rushed
                empathy=0.7,         # Understanding learning curve
                reassurance=0.7,     # "This is learnable"
                authority=0.6,       # Knowledgeable guide
                vocabulary_complexity="simple",
                sentence_structure="simple",
                explanation_depth="detailed",
                use_contractions=True,
                avoid_hype=True,
                focus_on_outcomes=True
            ),
            
            ToneProfile.NEUTRAL: ToneParameters(
                formality=0.5,       # Balanced CloudBridge baseline
                warmth=0.6,          # Friendly but professional
                energy=0.5,          # Steady and balanced
                directness=0.6,      # Clear and straightforward
                empathy=0.6,         # Understanding and respectful
                reassurance=0.5,     # Confident without overpromising
                authority=0.5,       # Balanced expertise
                vocabulary_complexity="moderate",
                sentence_structure="moderate",
                explanation_depth="standard",
                use_contractions=True,
                avoid_hype=True,
                focus_on_outcomes=True
            )
        }
    
    def _build_mapping_rules(self) -> Dict:
        """
        Map contexts to tone profiles
        Rules reflect CloudBridge approach to different situations
        """
        return {
            # Urgent, high-stakes situations
            ("urgent", "high_severity"): {
                "profile": ToneProfile.CALM_REASSURING,
                "adjustments": {"directness": 0.9, "energy": 0.3, "reassurance": 0.9}
            },
            
            # User is confused or struggling
            ("confused", "needs_help"): {
                "profile": ToneProfile.EDUCATIONAL,
                "adjustments": {"explanation_depth": "detailed", "empathy": 0.8}
            },
            
            # User is satisfied or grateful
            ("satisfied", "positive"): {
                "profile": ToneProfile.HELPFUL,
                "adjustments": {"warmth": 0.8, "energy": 0.6}
            },
            
            # User is frustrated or angry
            ("frustrated", "negative"): {
                "profile": ToneProfile.EMPATHETIC,
                "adjustments": {"empathy": 0.95, "reassurance": 0.8, "directness": 0.7}
            },
            
            # Professional business context
            ("formal", "business"): {
                "profile": ToneProfile.PROFESSIONAL,
                "adjustments": {"formality": 0.6}  # CloudBridge max
            },
            
            # Quick question, straightforward answer
            ("question", "simple"): {
                "profile": ToneProfile.SOLUTION_FOCUSED,
                "adjustments": {"directness": 0.8, "explanation_depth": "brief"}
            }
        }
    
    def select_tone(
        self,
        context: Dict,
        conversation_type: str,
        attention_indicators: Dict,
        conversation_history: Optional[List] = None
    ) -> ToneParameters:
        """
        Select the right tone for this moment in the conversation
        
        Args:
            context: Context analysis (emotion, urgency, technical level, etc.)
            conversation_type: Type of conversation (support, onboarding, etc.)
            attention_indicators: Signals needing attention (escalation, satisfaction)
            conversation_history: Previous turns for consistency
        
        Returns:
            ToneParameters validated against CloudBridge boundaries
        """
        
        # Choose base tone profile
        base_profile = self._select_base_profile(context, conversation_type, attention_indicators)
        
        # Get base parameters
        tone_params = self.tone_profiles[base_profile]
        
        # Apply context-specific adjustments
        tone_params = self._apply_contextual_adjustments(
            tone_params, context, attention_indicators
        )
        
        # Adjust for user preferences
        tone_params = self._apply_user_adjustments(tone_params, context)
        
        # Enforce CloudBridge brand boundaries (always)
        tone_params = self._enforce_brand_boundaries(tone_params)
        
        # Ensure smooth transitions if conversation is ongoing
        if conversation_history and len(conversation_history) > 0:
            tone_params = self._ensure_tone_consistency(tone_params)
        
        # Track for consistency
        self.tone_history.append(tone_params)
        
        return tone_params
    
    def _select_base_profile(
        self, 
        context: Dict, 
        conversation_type: str,
        attention_indicators: Dict
    ) -> ToneProfile:
        """Choose the right base tone profile"""
        
        emotion = context.get("emotion", "neutral")
        urgency = context.get("urgency_level", 2)
        frustration = context.get("frustration_level", 1)
        technical_level = context.get("technical_level", "intermediate")
        formality = context.get("formality_preference", "neutral")
        intent = context.get("intent", "question")
        
        # ====================================================================
        # PRIORITY 1: TECHNICAL USERS GET SOLUTION-FOCUSED RESPONSES
        # Technical users want answers, not empathy - even when frustrated
        # ====================================================================
        if technical_level in ["intermediate", "expert"]:
            # Technical problem reports → Direct solutions
            if intent in ["problem_report", "integration_help", "security_concern"]:
                return ToneProfile.SOLUTION_FOCUSED
            
            # Technical questions → Professional (clear, direct)
            if intent == "question":
                return ToneProfile.PROFESSIONAL
        
        # ====================================================================
        # PRIORITY 2: CRITICAL/URGENT SITUATIONS
        # Business-critical issues need calm confidence regardless of emotion
        # ====================================================================
        if urgency >= 4 or attention_indicators.get("needs_escalation", 0) > 0.7:
            return ToneProfile.CALM_REASSURING
        
        # Security issues always get professional, reassuring tone
        if conversation_type == "security_issue" or intent == "security_concern":
            return ToneProfile.CALM_REASSURING
        
        # ====================================================================
        # PRIORITY 3: EMOTIONAL STATE (only for non-technical users)
        # Empathy is important, but only when technical expertise isn't the priority
        # ====================================================================
        
        # ANGRY/HIGHLY FRUSTRATED non-technical users → Empathetic
        # (Technical users skip this - they want solutions, not empathy)
        if technical_level == "novice":
            if emotion in ["angry", "frustrated", "disappointed"] or frustration >= 4:
                return ToneProfile.EMPATHETIC
        
        # Moderately frustrated technical users → Solution-focused (not empathetic)
        # They're frustrated because something's broken, fix it
        if emotion in ["frustrated", "stressed"] and technical_level != "novice":
            return ToneProfile.SOLUTION_FOCUSED
        
        # ====================================================================
        # PRIORITY 4: LEARNING/GUIDANCE NEEDS
        # ====================================================================
        
        # Confused or new users → Educational
        if emotion == "confused" or technical_level == "novice":
            return ToneProfile.EDUCATIONAL
        
        # Onboarding conversations → Educational
        if conversation_type == "onboarding":
            return ToneProfile.EDUCATIONAL
        
        # ====================================================================
        # PRIORITY 5: POSITIVE INTERACTIONS
        # ====================================================================
        
        # Satisfied customers → Helpful (warm, supportive)
        if emotion in ["satisfied", "grateful", "relieved"]:
            return ToneProfile.HELPFUL
        
        # ====================================================================
        # PRIORITY 6: TIME-SENSITIVE SITUATIONS
        # ====================================================================
        
        # Time-critical but not urgent enough for CALM_REASSURING → Solution-Focused
        if context.get("time_sensitivity", False) and urgency >= 3:
            return ToneProfile.SOLUTION_FOCUSED
        
        # ====================================================================
        # PRIORITY 7: FORMALITY PREFERENCES
        # ====================================================================
        
        # Executives, senior professionals → Professional
        if formality == "formal" or context.get("inferred_age_group") == "executive":
            return ToneProfile.PROFESSIONAL
        
        # ====================================================================
        # PRIORITY 8: CONVERSATION TYPE FALLBACKS
        # ====================================================================
        
        # Complaint resolution → Start empathetic for non-technical
        if conversation_type == "complaint_resolution" and technical_level == "novice":
            return ToneProfile.EMPATHETIC
        
        # Technical implementation → Professional
        if conversation_type == "technical_implementation":
            return ToneProfile.PROFESSIONAL
        
        # ====================================================================
        # DEFAULT: HELPFUL (CloudBridge's natural state)
        # Professional but approachable, warm but efficient
        # ====================================================================
        return ToneProfile.HELPFUL
    
    def _apply_contextual_adjustments(
        self,
        tone_params: ToneParameters,
        context: Dict,
        attention_indicators: Dict
    ) -> ToneParameters:
        """Fine-tune tone based on specific context"""
        
        # Create adjusted copy
        adjusted = ToneParameters(**tone_params.__dict__)
        
        # Urgency adjustments
        urgency = context.get("urgency_level", 2)
        if urgency >= 4:
            adjusted.directness = min(1.0, adjusted.directness + 0.2)
            adjusted.energy = max(0.2, adjusted.energy - 0.2)  # Stay calm
            adjusted.explanation_depth = "brief"  # Get to solutions quickly
        
        # Declining satisfaction → more empathy and reassurance
        if attention_indicators.get("satisfaction_declining", 0) > 0.5:
            adjusted.empathy = min(1.0, adjusted.empathy + 0.3)
            adjusted.reassurance = min(1.0, adjusted.reassurance + 0.3)
            adjusted.warmth = min(1.0, adjusted.warmth + 0.2)
        
        # Time-sensitive → more direct
        if context.get("time_sensitivity", False):
            adjusted.directness = min(1.0, adjusted.directness + 0.2)
            adjusted.explanation_depth = "brief"
        
        # Technical expertise adjustments
        tech_level = context.get("technical_level", "intermediate")
        if tech_level == "expert":
            adjusted.vocabulary_complexity = "technical"
            adjusted.explanation_depth = "brief"  # They know the basics
        elif tech_level == "novice":
            adjusted.vocabulary_complexity = "simple"
            adjusted.explanation_depth = "detailed"
        
        return adjusted
    
    def _apply_user_adjustments(
        self,
        tone_params: ToneParameters,
        context: Dict
    ) -> ToneParameters:
        """Adjust tone for user preferences and characteristics"""
        
        age_group = context.get("inferred_age_group", "mid_career")
        adjusted = ToneParameters(**tone_params.__dict__)
        
        if age_group == "early_career":
            # Slightly more casual, but still professional
            adjusted.formality = max(0.4, adjusted.formality - 0.1)
            adjusted.energy = min(0.8, adjusted.energy + 0.1)
            adjusted.vocabulary_complexity = "simple"
            
        elif age_group in ["senior_professional", "executive"]:
            # Maintain professionalism, but still use contractions
            adjusted.formality = min(0.7, adjusted.formality + 0.1)  # CloudBridge max
            adjusted.warmth = min(1.0, adjusted.warmth + 0.1)
            adjusted.explanation_depth = "standard"  # Respect their time
            # Still use contractions - CloudBridge brand requirement
            
        elif age_group == "mid_career":
            # CloudBridge baseline - efficient and professional
            adjusted.formality = 0.5
            adjusted.directness = min(0.8, adjusted.directness + 0.1)
        
        return adjusted
    
    def _enforce_brand_boundaries(self, tone_params: ToneParameters) -> ToneParameters:
        """
        Enforce CloudBridge brand tone boundaries
        Never compromise on brand voice, regardless of context
        """
        
        boundaries = self.brand_boundaries
        adjusted = ToneParameters(**tone_params.__dict__)
        
        # CloudBridge formality range: 0.4 (approachable) to 0.7 (professional)
        max_formality = boundaries.get("max_formality", 0.7)
        min_formality = boundaries.get("min_formality", 0.4)
        adjusted.formality = max(min_formality, min(adjusted.formality, max_formality))
        
        # CloudBridge is always warm and helpful (min 0.5)
        min_warmth = boundaries.get("min_warmth", 0.5)
        adjusted.warmth = max(adjusted.warmth, min_warmth)
        
        # CloudBridge is direct and clear (min 0.6)
        min_directness = boundaries.get("min_directness", 0.6)
        adjusted.directness = max(adjusted.directness, min_directness)
        
        # CloudBridge never uses hype or pressure
        adjusted.avoid_hype = True
        
        # Always focus on outcomes and enablement
        adjusted.focus_on_outcomes = True
        
        # Moderate contractions for approachability
        adjusted.use_contractions = True
        
        return adjusted
    
    def _ensure_tone_consistency(self, new_tone: ToneParameters) -> ToneParameters:
        """
        Smooth tone transitions across conversation
        Prevents jarring shifts that break rapport
        """
        
        if not self.tone_history:
            return new_tone
        
        previous_tone = self.tone_history[-1]
        adjusted = ToneParameters(**new_tone.__dict__)
        
        # Prevent large shifts (max 0.3 change per parameter)
        MAX_SHIFT = 0.3
        
        adjusted.formality = self._smooth_transition(
            previous_tone.formality, new_tone.formality, MAX_SHIFT
        )
        adjusted.warmth = self._smooth_transition(
            previous_tone.warmth, new_tone.warmth, MAX_SHIFT
        )
        adjusted.energy = self._smooth_transition(
            previous_tone.energy, new_tone.energy, MAX_SHIFT
        )
        adjusted.empathy = self._smooth_transition(
            previous_tone.empathy, new_tone.empathy, MAX_SHIFT
        )
        
        # Brand requirements never change
        adjusted.use_contractions = True
        adjusted.avoid_hype = True
        adjusted.focus_on_outcomes = True
        
        return adjusted
    
    def _smooth_transition(self, old_value: float, new_value: float, max_shift: float) -> float:
        """Gradually shift tone values to avoid jarring changes"""
        diff = new_value - old_value
        
        if abs(diff) <= max_shift:
            return new_value
        
        # Limit the shift
        if diff > 0:
            return old_value + max_shift
        else:
            return old_value - max_shift
    
    def get_generation_instructions(self, tone_params: ToneParameters, context: Dict) -> str:
        """
        Generate clear instructions for response generation
        Translates tone parameters into actionable guidance
        """
        
        instructions = []
        
        # CloudBridge brand voice (always first)
        instructions.append("Write in CloudBridge voice: clear, outcome-focused, professional yet approachable.")
        instructions.append("Use contractions naturally (I'm, you're, don't, can't, here's, that's).")
        instructions.append("Never use hype language, pressure tactics, or buzzwords.")
        
        # Formality
        if tone_params.formality > 0.6:
            instructions.append("Maintain professional tone while staying conversational.")
        elif tone_params.formality < 0.45:
            instructions.append("Use approachable, friendly language.")
        else:
            instructions.append("Balance professionalism with approachability.")
        
        # Warmth & Empathy
        if tone_params.empathy > 0.7:
            instructions.append("Show genuine empathy and understanding.")
        if tone_params.warmth > 0.7:
            instructions.append("Be warm and supportive.")
        
        # Energy & Directness
        if tone_params.energy < 0.3:
            instructions.append("Stay calm and measured.")
        if tone_params.directness > 0.7:
            instructions.append("Be direct and solution-focused.")
        
        # Reassurance
        if tone_params.reassurance > 0.7:
            instructions.append("Reassure confidently without overpromising.")
        
        # Complexity
        if tone_params.vocabulary_complexity == "simple":
            instructions.append("Use clear, jargon-free language.")
        elif tone_params.vocabulary_complexity == "technical":
            instructions.append("Technical terms are fine if relevant.")
        
        # Explanation depth
        if tone_params.explanation_depth == "detailed":
            instructions.append("Provide step-by-step explanations.")
        elif tone_params.explanation_depth == "brief":
            instructions.append("Keep it concise. Get to solutions quickly.")
        # Formatting preferences
        if not tone_params.use_bold_text:
            instructions.append("Do not use markdown bold (**text**).")
        
        if tone_params.preferred_bullet_style == "-":
            instructions.append("Use simple dash bullets (-).")
        elif tone_params.preferred_bullet_style == "•":
            instructions.append("Use bullet points (•).")
        
        # Outcome focus
        instructions.append("Focus on what the user can accomplish, not just features.")
        
        return " ".join(instructions)
    
    def reset_history(self):
        """Reset tone history (for new conversation sessions)"""
        self.tone_history = []