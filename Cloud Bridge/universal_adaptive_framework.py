"""
Universal Adaptive Framework - CloudBridge Edition
Integrates all components for brand-consistent, context-aware, adaptive AI
Enhanced with CloudBridge brand DNA enforcement and validation retry logic
CloudBridge Technology Solutions
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from brand_consistency_guard import BrandConsistencyGuard
from conversation_state_manager import ConversationStateManager
from context_understanding_engine import ContextUnderstandingEngine
from tone_adaptation_engine import ToneAdaptationEngine
from performance_tracker import PerformanceTracker, StepTimer
from model_optimizer import ModelOptimizer, initialize_optimizer, OptimizationStrategy

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize global components
tracker = PerformanceTracker()
optimizer = initialize_optimizer(client, strategy=OptimizationStrategy.BALANCED)

# CloudBridge Brand Configuration
BRAND_ETHOS = {
    "name": "CloudBridge",
    "tagline": "Empower Every Person",
    "core_values": ["Inclusivity", "Innovation", "Trust", "Collaboration"],
    "personality": "The reliable partner who helps you accomplish what matters most. Empowering without being patronizing, professional without being cold, innovative without being flashy.",
    "communication_style": "Clear, jargon-free, outcome-focused. Professional yet approachable. Natural and conversational with moderate contractions.",
    "voice_guidelines": {
        "do": [
            "Use clear, jargon-free language",
            "Focus on outcomes, not just features",
            "Keep it accessible and straightforward",
            "Acknowledge real challenges, offer real solutions",
            "Make it inclusive and empowering",
            "Use natural, conversational tone with contractions",
            "Enable, don't prescribe",
            "Be specific and actionable",
            "Show genuine empathy when appropriate"
        ],
        "dont": [
            "Use technical jargon or acronym overload (synergy, leverage, ecosystem, paradigm)",
            "Use hype or exaggeration (revolutionary, incredible, game-changing)",
            "Apply pressure or urgency tactics (act now, limited time, don't miss out)",
            "Sound condescending or oversimplified (even you can, don't worry it's easy)",
            "Make vague promises without substance (transform your business, experience the future)",
            "Use corporate template language (thank you for contacting, is there anything else)",
            "Over-apologize or sound robotic"
        ]
    }
}

@dataclass
class FrameworkResponse:
    """Complete framework response with all metadata"""
    response_text: str
    context_analysis: Dict
    tone_used: Dict
    brand_validation: Dict
    conversation_state: Dict
    performance_metrics: Dict
    should_escalate: bool
    confidence: float

class UniversalAdaptiveFramework:
    """
    Production-ready adaptive framework with CloudBridge brand enforcement
    
    What it does:
    - Enforces CloudBridge brand consistency across all responses
    - Understands context and maintains conversation memory
    - Adapts tone within CloudBridge boundaries
    - Assesses risk and recommends escalation when needed
    - Optimizes performance with model selection
    - Auto-retries responses that don't meet brand standards
    
    Why it matters:
    Every interaction represents CloudBridge. This ensures consistency,
    quality, and brand alignment at scale.
    """
    
    def __init__(self, brand_config: Dict, client: OpenAI):
        """
        Initialize framework with all components
        
        Args:
            brand_config: CloudBridge brand configuration (BRAND_ETHOS)
            client: OpenAI client
        """
        self.client = client
        self.brand_config = brand_config
        
        # Initialize core components
        self.brand_guard = BrandConsistencyGuard(brand_config, client, threshold=0.75)
        self.context_engine = ContextUnderstandingEngine(client)
        self.tone_adapter = ToneAdaptationEngine(
            self.brand_guard.get_boundaries(), 
            client
        )
        
        # Session management
        self.active_sessions: Dict[str, ConversationStateManager] = {}
        
        # Scope validation patterns (CloudBridge product support)
        self.in_scope_keywords = [
            'cloudbridge', 'account', 'billing', 'subscription', 'plan',
            'sync', 'collaboration', 'team', 'workspace', 'document', 'file',
            'integration', 'api', 'security', 'privacy', 'password',
            'video', 'call', 'meeting', 'calendar', 'schedule',
            'storage', 'backup', 'recover', 'export', 'import',
            'crash', 'bug', 'error', 'issue', 'problem', 'not working',
            'feature', 'settings', 'help', 'how to', 'can you', 'urgent',
            'migration', 'setup', 'configuration'
        ]
        
    def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationStateManager:
        """Get existing session or create new one"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationStateManager(session_id, user_id)
        return self.active_sessions[session_id]
    
    def process_message(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> FrameworkResponse:
        """
        Process user message through complete CloudBridge framework
        
        Args:
            message: User's message
            session_id: Unique session identifier
            user_id: Optional user identifier
        
        Returns:
            FrameworkResponse with response and comprehensive metadata
        """
        
        tracker.start_pipeline(f"Session: {session_id}")
        
        # 1. Get or create conversation state
        conv_state = self.get_or_create_session(session_id, user_id)
        
        # 2. Validate scope (is this a CloudBridge-related question?)
        is_in_scope, redirect_msg = self._validate_scope(message, conv_state)
        if not is_in_scope:
            return self._create_out_of_scope_response(redirect_msg, conv_state)
        
        # 3. Deep context analysis (HYBRID: pattern OR AI)
        context_start = time.time()
        context_analysis = self.context_engine.analyze(
            message=message,
            history=conv_state.turns,
            user_profile=conv_state.user_profile.__dict__
        )
        
        # Track tokens if AI was used
        context_tokens = getattr(context_analysis, 'tokens_used', 0)
        used_ai = getattr(context_analysis, 'used_ai', False)
        if context_tokens > 0 and used_ai:
            # Create metrics for API call with proper timing
            step_metrics = tracker.track_api_call(
                step_name="Context Analysis",
                model="gpt-4o-mini",
                response=type('obj', (object,), {
                    'usage': type('obj', (object,), {
                        'prompt_tokens': int(context_tokens * 0.6),
                        'completion_tokens': int(context_tokens * 0.4),
                        'total_tokens': context_tokens
                    })()
                })(),
                start_time=context_start
            )
        else:
            # Non-API step (pattern matching)
            step_metrics = tracker.track_non_api_step("Context Analysis")
            step_metrics.start_time = context_start
            step_metrics.duration = time.time() - context_start
        
        # 4. Get conversation state summary
        conv_state_summary = conv_state.get_context_summary()
        
        # 5. Determine conversation type
        conversation_type = context_analysis.conversation_type.value
        
        # 6. Risk assessment (renamed from "risk_flags" to "attention_indicators")
        attention_indicators = {
            "needs_escalation": conv_state.needs_escalation,
            "satisfaction_declining": conv_state.satisfaction_declining,
            "repeated_frustration": conv_state.repeated_frustration_count
        }
        
        # Check if should escalate
        should_escalate = conv_state.should_escalate()
        
        # 7. Select tone within CloudBridge boundaries (non-API step - pure rule-based)
        tone_start = time.time()
        tone_params = self.tone_adapter.select_tone(
            context=context_analysis.to_dict(),
            conversation_type=conversation_type,
            attention_indicators=attention_indicators,
            conversation_history=conv_state.turns
        )
        
        # Tone selection is pure rule-based (no API)
        step_metrics = tracker.track_non_api_step("Tone Selection")
        step_metrics.start_time = tone_start
        step_metrics.duration = time.time() - tone_start
        
        # 8. Generate brand-validated response with auto-retry (API call)
        with StepTimer(tracker, "Adaptive Response Generation (with validation)", is_api_call=True) as timer:
            response_text, brand_validation_dict, response_metrics = self._generate_brand_validated_response(
                message=message,
                context=context_analysis,
                tone=tone_params,
                conv_state=conv_state,
                should_escalate=should_escalate,
                max_attempts=3
            )
            if response_metrics:
                timer.set_metrics(response_metrics)
        
        # Convert dict to object-like structure for compatibility
        class BrandValidationResult:
            def __init__(self, d):
                self.__dict__.update(d)
                self.passed = d.get('passed', False)
                self.overall_score = d.get('overall_score', 0.0)
                self.violations = d.get('violations', [])
        
        brand_validation = BrandValidationResult(brand_validation_dict)
        
        # 9. Update conversation state with this turn
        conv_state.add_turn(
            user_message=message,
            assistant_response=response_text,
            context=context_analysis.to_dict(),
            tone=tone_params.to_dict()
        )
        
        # 10. Calculate confidence score
        confidence = self._calculate_response_confidence(
            brand_score=brand_validation.overall_score,
            context_confidence=context_analysis.confidence,
            tone_appropriateness=1.0
        )
        
        # 11. Build complete response
        performance = tracker.end_pipeline()
        
        return FrameworkResponse(
            response_text=response_text,
            context_analysis=context_analysis.to_dict(),
            tone_used=tone_params.to_dict(),
            brand_validation={
                "passed": brand_validation.passed,
                "overall_score": brand_validation.overall_score,
                "scores": brand_validation_dict.get('scores', {}),
                "violations_count": len(brand_validation.violations),
                "feedback": brand_validation_dict.get('detailed_feedback', '')
            },
            conversation_state=conv_state_summary,
            performance_metrics=performance.to_dict() if performance else {},
            should_escalate=should_escalate,
            confidence=confidence
        )
    
    def _validate_scope(
        self, 
        message: str, 
        conv_state: ConversationStateManager
    ) -> Tuple[bool, Optional[str]]:
        """Validate if message is in CloudBridge product scope"""
        
        msg_lower = message.lower()
        
        # Fast-path: obvious in-scope keywords
        if any(kw in msg_lower for kw in self.in_scope_keywords):
            return True, None
        
        # If ongoing conversation, be generous with scope
        if len(conv_state.turns) > 0:
            return True, None
        
        # Obvious out-of-scope topics
        out_of_scope = ['weather', 'joke', 'story', 'poem', 'recipe', 'capital of', 'who won']
        if any(pattern in msg_lower for pattern in out_of_scope):
            return False, "I'm here to help with CloudBridge. How can I help with your workspace today?"
        
        # For ambiguous cases, default to in-scope
        return True, None
    
    def _generate_brand_validated_response(
        self,
        message: str,
        context,
        tone,
        conv_state: ConversationStateManager,
        should_escalate: bool,
        max_attempts: int = 3
    ) -> Tuple[str, Dict, Optional[object]]:
        """
        Generate response with automatic CloudBridge brand validation
        Retries if response doesn't meet brand standards
        
        Returns:
            (response_text, brand_validation_dict, metrics)
        """
        
        previous_violations = []
        first_metrics = None  # Always return the initial response's metrics to the caller
        
        for attempt in range(max_attempts):
            # Generate or regenerate response
            if attempt == 0:
                response_text, metrics = self._generate_response(
                    message, context, tone, conv_state, should_escalate
                )
                first_metrics = metrics  # Capture before any retry can overwrite
            else:
                print(f"🔄 Regenerating with CloudBridge brand constraints (attempt {attempt + 1}/{max_attempts})...")
                response_text, metrics = self._regenerate_with_constraints(
                    message, context, tone, previous_violations
                )
            
            # Validate against CloudBridge brand guard
            brand_validation = self.brand_guard.validate(
                response=response_text,
                context=context.to_dict(),
                conversation_type=context.conversation_type.value
            )
            
            # Check if passed CloudBridge standards
            if brand_validation.passed:
                print(f"CloudBridge brand validation passed on attempt {attempt + 1} (score: {brand_validation.overall_score:.2%})")
                return response_text, {
                    'passed': True,
                    'overall_score': brand_validation.overall_score,
                    'scores': brand_validation.scores,
                    'violations': [],
                    'detailed_feedback': brand_validation.detailed_feedback
                }, first_metrics
            
            # Failed - log and prepare for retry
            print(f" Attempt {attempt + 1}/{max_attempts} - Brand score: {brand_validation.overall_score:.2%}")
            print(f"   Violations: {len(brand_validation.violations)}")
            for v in brand_validation.violations[:3]:
                print(f"   - {v.description}")
            
            previous_violations = brand_validation.violations
        
        print(f" Failed brand validation after {max_attempts} attempts (final score: {brand_validation.overall_score:.2%})")
        return response_text, {
            'passed': False,
            'overall_score': brand_validation.overall_score,
            'scores': brand_validation.scores,
            'violations': [
                {'severity': v.severity, 'category': v.category, 'description': v.description}
                for v in brand_validation.violations
            ],
            'detailed_feedback': brand_validation.detailed_feedback
        }, first_metrics
    
    def _generate_response(
        self,
        message: str,
        context,
        tone,
        conv_state: ConversationStateManager,
        should_escalate: bool
    ) -> Tuple[str, Optional[object]]:
        """Generate response using LLM - returns (response_text, metrics)"""
        
        gen_start = time.time()
        
        # Build system prompt with CloudBridge brand DNA
        system_prompt = self._build_system_prompt(tone, context, should_escalate)
        
        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent history (last 3 turns for context)
        for turn in conv_state.turns[-3:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_response})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Select optimal model
        selected_model = optimizer.select_model(
            step_name="Adaptive Response Generation",
            prompt=message,
            estimated_input_tokens=sum(len(m['content'].split()) for m in messages) * 1.3,
            estimated_output_tokens=350
        )
        
        # Generate response
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=0.7,
            max_tokens=350
        )
        
        # Track metrics with proper start time
        metrics = tracker.track_api_call("Adaptive Response Generation", selected_model, response, start_time=gen_start)
        
        return response.choices[0].message.content.strip(), metrics
    
    def _build_system_prompt(self, tone, context, should_escalate: bool) -> str:
        """Build system prompt with CloudBridge brand DNA and tone instructions"""
        
        tone_instructions = self.tone_adapter.get_generation_instructions(tone, context.to_dict())
        
        prompt = f"""You are a CloudBridge support specialist. CloudBridge is a technology company that sounds like a REAL PERSON helping someone, not a corporate chatbot.

══════════════════════════════════════════
HARD RULES — THESE OVERRIDE EVERYTHING ELSE:
══════════════════════════════════════════
1. MAX 80 WORDS. Count every word. Stop at 80.
2. NO BULLET POINTS. NO NUMBERED LISTS. NO DASHES AS LIST MARKERS.
3. NO "First... Then... Finally/Lastly" — prose only, like a colleague speaks.
4. NO COMPETITOR NAMES — never say AWS, Azure, Google Cloud, Microsoft, Salesforce, etc.
5. NO OPEN-ENDED CLOSERS — never end with "Let me know if...", "Need help with...?", "If you want...", "Is there anything else...?"
6. NO FILLER OPENERS — never start with "Great!", "Absolutely!", "Of course!", "Sure!", "Certainly!"
7. CONTRACTIONS ARE MANDATORY — I'll, you're, it's, don't, can't, here's, we'll
8. BE SPECIFIC — never say "a few weeks", "some steps", "various options". Name real things.
══════════════════════════════════════════

     YOUR #1 RULE: Sound like a helpful colleague, NOT a customer service bot.

     WRITE LIKE THIS (CloudBridge Voice):
    "Here's what's happening with your sync issue. Your files are stuck in the queue. Let me get that moving for you."

    "That billing charge? It's from the annual renewal on Dec 1st. I can break down the details if you'd like."

    "Getting a 500 error on POST /api/v2/users? Check your auth token first—that's usually the culprit. Still seeing it after that?"

     NEVER WRITE LIKE THIS (Corporate Bot):
    "Thank you for contacting CloudBridge support. I understand your frustration regarding this matter. I will be happy to assist you with this issue."

    "I apologize for any inconvenience this may have caused. Is there anything else I can help you with today?"

     CLOUDBRIDGE DNA (USE THESE IN EVERY RESPONSE):
    1. **Contractions are MANDATORY**: I'll, you're, it's, that's, don't, can't, here's
    -  "I will help you" →  "I'll help you"
    -  "Here is what is happening" →  "Here's what's happening"

    2. **Get to the point fast**: No preamble, no fluff
    -  "Thank you for reaching out. I understand you're experiencing an issue."
    -  "Let's fix that sync issue."

    3. **Be specific, not vague**:
    -  "We can help you with that."
    -  "Go to Settings > Team > Sync Options."

    4. **Sound human, not scripted**:
    -  "Is there anything else I can assist you with?"
    -  "Need anything else?" (or just end naturally)

    5. **Show you understand without being condescending**:
    -  "I can understand how frustrating that must be for you."
    -  "That's frustrating. Let's sort it out."

    TONE ADAPTATION FOR THIS MESSAGE:
    {tone_instructions}

    USER CONTEXT:
    - Emotion: {context.primary_emotion} (intensity: {context.emotion_intensity}/5)
    - Technical Level: {context.technical_level}
    - Urgency: {context.urgency_level}/5"""

        # Add technical user guidance
        if context.technical_level in ["intermediate", "expert"]:
            prompt += """

     TECHNICAL USER DETECTED:
    - Start with the solution immediately
    - Use technical terms appropriately (API, endpoint, auth token, etc.)
    - Skip long explanations—they want the fix
    - Example: "500 on POST /api/v2/users. Check your auth header. Should be 'Bearer <token>'. Still hitting it?"
    """
        else:
            prompt += """

     NON-TECHNICAL USER:
    - Break it down simply
    - Avoid jargon
    - Be patient and encouraging
    - Example: "Let's get you set up. First, click Settings in the top right..."
    """

        prompt += f"""

    NOW RESPOND TO THIS MESSAGE:
    "{context.primary_intent.value}" intent detected.
FORMATTING RULES - CRITICAL:
- When listing steps, put TWO line breaks before and after the list
- Put ONE line break between each bullet point.

    REMEMBER: Sound like a real person, not a chatbot. Use contractions. Be direct. Be helpful."""

        return prompt
    
    def _get_cloudbridge_voice_examples(self) -> str:
        """Get CloudBridge brand voice examples for system prompt"""
        return """
CLOUDBRIDGE VOICE EXAMPLES:

 BAD (Corporate/Hype):
"Thank you for contacting CloudBridge! We're excited to help you leverage our revolutionary collaboration platform to synergize your team's productivity. Don't miss this incredible opportunity!"

 GOOD (CloudBridge Brand):
"Here's how to set that up. Go to Settings, then Team Collaboration. You'll see the option to connect your calendar. Once that's done, your team can schedule meetings without the back-and-forth emails."

---

 BAD: "CloudBridge provides a best-in-class solution for enterprise collaboration."
 GOOD: "CloudBridge connects your team. Work together from anywhere."

---

 BAD: "Please don't hesitate to reach out if you require additional assistance."
 GOOD: "Need anything else? I'm here to help." OR simply end naturally

---

 BAD: "I apologize for any inconvenience this may have caused to your organization."
 GOOD: "Sorry about that. Let's get this fixed for you."

---

 BAD: "Our cutting-edge technology leverages cloud-native infrastructure."
 GOOD: "Everything's secure in the cloud. Access your work from any device."

REMEMBER: Professional but not cold. Helpful but not patronizing. Clear but not condescending.
"""
    
    def _regenerate_with_constraints(
        self,
        message: str,
        context,
        tone,
        violations: List
    ) -> Tuple[str, Optional[object]]:
        """
        Regenerate response with stricter CloudBridge brand constraints
        Returns (response_text, metrics)
        """
        
        # Categorize violations
        forbidden_phrase_violations = [v for v in violations if 'forbidden' in v.category.lower()]
        linguistic_violations = [v for v in violations if 'linguistic' in v.category.lower() or 'dna' in v.category.lower()]
        tone_violations = [v for v in violations if 'tone' in v.category.lower()]
        clarity_violations = [v for v in violations if 'clarity' in v.category.lower()]
        
        # Build specific constraints
        constraint_notes = []
        
        if forbidden_phrase_violations:
            constraint_notes.append("🚨 FORBIDDEN PHRASES DETECTED - Remove immediately:")
            for v in forbidden_phrase_violations[:3]:
                constraint_notes.append(f"    {v.description}")
                if v.suggestion:
                    constraint_notes.append(f"    {v.suggestion}")
        
        if linguistic_violations:
            constraint_notes.append("\n🚨 LINGUISTIC DNA VIOLATIONS - Fix these:")
            for v in linguistic_violations[:3]:
                constraint_notes.append(f"    {v.description}")
                if v.suggestion:
                    constraint_notes.append(f"    {v.suggestion}")
        
        if clarity_violations:
            constraint_notes.append("\n🚨 CLARITY VIOLATIONS - Simplify:")
            for v in clarity_violations[:2]:
                constraint_notes.append(f"    {v.description}")
        
        # Add explicit CloudBridge requirements
        constraint_notes.append("\n CLOUDBRIDGE MANDATORY REQUIREMENTS:")
        constraint_notes.append("   1. USE CONTRACTIONS: I'm, you're, don't, can't, it's, here's")
        constraint_notes.append("   2. NO CORPORATE SPEAK: Never 'Thank you for contacting', 'Is there anything else'")
        constraint_notes.append("   3. PROFESSIONAL YET APPROACHABLE: Not cold, not overly casual")
        constraint_notes.append("   4. CLEAR LANGUAGE: No buzzwords, no jargon unless necessary")
        constraint_notes.append("   5. OUTCOME-FOCUSED: Tell them what they can accomplish")
        constraint_notes.append("   6. SENTENCE LENGTH: 8-18 words average")
        
        system_prompt = f"""You are a {self.brand_config['name']} support specialist.

 PREVIOUS RESPONSE VIOLATED CLOUDBRIDGE BRAND VOICE - REWRITE NOW:

{chr(10).join(constraint_notes)}

CloudBridge Values: {', '.join(self.brand_config['core_values'])}
CloudBridge Personality: {self.brand_config['personality']}

USER MESSAGE: "{message}"

USER CONTEXT: {context.primary_emotion} emotion, {context.urgency_level}/5 urgency

CRITICAL INSTRUCTIONS FOR REWRITE:
- STRICTLY follow CloudBridge's brand voice
- Use natural contractions throughout (I'm, don't, can't, it's, here's)
- Sound professional yet approachable
- Avoid ALL template phrases and corporate buzzwords
- Keep it clear,concise, specific, and outcome-focused
- Show appropriate empathy without over-apologizing
- Focus on what the user can accomplish

Generate the CORRECTED CloudBridge response NOW."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        regen_start = time.time()
        
        selected_model = optimizer.select_model(
            step_name="Constrained Regeneration",
            prompt=message,
            estimated_output_tokens=350
        )
        
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=0.6,  # Slightly lower for brand consistency
            max_tokens=350
        )
        
        metrics = tracker.track_api_call("Constrained Regeneration", selected_model, response, start_time=regen_start)
        
        return response.choices[0].message.content.strip(), metrics
    
    def _calculate_response_confidence(
        self,
        brand_score: float,
        context_confidence: float,
        tone_appropriateness: float
    ) -> float:
        """Calculate overall confidence in response quality"""
        weights = {
            "brand": 0.4,      # Brand consistency is most important
            "context": 0.3,    # Understanding context
            "tone": 0.3        # Appropriate tone
        }
        
        confidence = (
            brand_score * weights["brand"] +
            context_confidence * weights["context"] +
            tone_appropriateness * weights["tone"]
        )
        
        return round(confidence, 3)
    
    def _create_out_of_scope_response(
        self,
        redirect_msg: str,
        conv_state: ConversationStateManager
    ) -> FrameworkResponse:
        """Create response for out-of-scope messages"""
        
        return FrameworkResponse(
            response_text=redirect_msg,
            context_analysis={"out_of_scope": True},
            tone_used={},
            brand_validation={"passed": True, "overall_score": 1.0, "scores": {}, "violations_count": 0, "feedback": "Out of scope"},
            conversation_state=conv_state.get_context_summary(),
            performance_metrics={},
            should_escalate=False,
            confidence=1.0
        )
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get conversation summary for a session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].to_dict()
        return None
    
    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


# ============================================================================
# DEMO / TESTING CODE
# ============================================================================

def run_framework_demo():
    """Demo of the complete CloudBridge-enhanced framework"""
    
    print("\n" + "="*80)
    print(" CLOUDBRIDGE UNIVERSAL ADAPTIVE FRAMEWORK")
    print("   Empower Every Person")
    print("="*80)
    
    # Initialize framework
    framework = UniversalAdaptiveFramework(BRAND_ETHOS, client)
    
    # Test scenarios that reflect CloudBridge use cases
    test_scenarios = [
        {
            "session_id": "test_001",
            "messages": [
                "I'm having trouble getting our team synced up on CloudBridge. Files aren't appearing for everyone.",
            ]
        },
        {
            "session_id": "test_002",
            "messages": [
                "This is really frustrating. Our video calls keep dropping and we have an important client meeting in an hour!",
            ]
        },
        {
            "session_id": "test_003",
            "messages": [
                "Quick question about the API integration with our CRM system",
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"📋 SESSION: {scenario['session_id']}")
        print(f"{'='*80}\n")
        
        for i, message in enumerate(scenario['messages'], 1):
            print(f"Turn {i}:")
            print(f"👤 User: {message}")
            
            # Process message through framework
            response = framework.process_message(
                message=message,
                session_id=scenario['session_id']
            )
            
            print(f" CloudBridge: {response.response_text}")
            print(f"\n Metadata:")
            print(f"   - Emotion: {response.context_analysis.get('emotion', 'N/A')}")
            print(f"   - Intent: {response.context_analysis.get('intent', 'N/A')}")
            print(f"   - Brand Score: {response.brand_validation['overall_score']:.2%} {'' if response.brand_validation['passed'] else ''}")
            print(f"   - Confidence: {response.confidence:.2%}")
            if response.brand_validation['violations_count'] > 0:
                print(f"   - ⚠️ Brand Violations: {response.brand_validation['violations_count']}")
            if response.should_escalate:
                print(f"   - ⚠️ ESCALATION RECOMMENDED")
            print()
    
    # Performance summary
    print("\n" + "="*80)
    print(" PERFORMANCE SUMMARY")
    print("="*80)
    tracker.print_summary()
    
    print("\n CloudBridge framework demo complete!")

if __name__ == "__main__":
    run_framework_demo()