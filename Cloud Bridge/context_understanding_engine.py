"""
Context Understanding Engine - HYBRID CloudBridge Edition
Pattern matching first, AI fallback with proper token tracking
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from openai import OpenAI

class Intent(Enum):
    """User intent categories"""
    PROBLEM_REPORT = "problem_report"
    QUESTION = "question"
    COMPLAINT = "complaint"
    FEATURE_REQUEST = "feature_request"
    PRAISE = "praise"
    CLARIFICATION = "clarification"
    CHURN_SIGNAL = "churn_signal"
    SMALL_TALK = "small_talk"
    CLOSING = "closing"
    ESCALATION_REQUEST = "escalation_request"
    INTEGRATION_HELP = "integration_help"
    SECURITY_CONCERN = "security_concern"
    MIGRATION_INQUIRY = "migration_inquiry"

class ConversationType(Enum):
    """Types of conversations"""
    SUPPORT = "support"
    SALES = "sales"
    ONBOARDING = "onboarding"
    FEEDBACK = "feedback"
    COMPLAINT_RESOLUTION = "complaint_resolution"
    CHITCHAT = "chitchat"
    EMERGENCY = "emergency"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    SECURITY_ISSUE = "security_issue"

@dataclass
class ContextAnalysis:
    """Comprehensive context analysis result"""
    # Core emotion & sentiment
    primary_emotion: str
    emotion_intensity: int  # 1-5
    secondary_emotions: List[str]
    sentiment_score: float  # -1 to 1
    
    # Intent & purpose
    primary_intent: Intent
    confidence: float
    multi_intent: bool
    intent_details: Dict
    
    # Conversation type
    conversation_type: ConversationType
    
    # User state
    user_state: str  # "calm", "concerned", "frustrated", etc.
    urgency_level: int  # 1-5
    frustration_level: int  # 1-5
    
    # Communication style
    formality_preference: str  # "casual", "neutral", "formal"
    directness: float  # 0-1, how direct user is
    verbosity: str  # "brief", "moderate", "verbose"
    
    # Demographics & culture
    inferred_age_group: Optional[str]
    age_confidence: str  # "low", "medium", "high"
    age_indicators: List[str]
    cultural_signals: List[str]
    
    # Technical & expertise
    technical_level: str  # "novice", "intermediate", "expert"
    domain_knowledge: str  # How much they know about the product
    
    # Context & situation
    situation_type: str
    key_pain_points: List[str]
    mentioned_features: List[str]
    time_sensitivity: bool
    
    # Risk indicators
    escalation_indicators: List[str]
    churn_indicators: List[str]
    satisfaction_indicators: List[str]
    
    # HYBRID TRACKING
    tokens_used: int = 0      
    used_ai: bool = False     
    
    def to_dict(self) -> Dict:
        return {
            "emotion": self.primary_emotion,
            "emotion_intensity": self.emotion_intensity,
            "secondary_emotions": self.secondary_emotions,
            "sentiment_score": self.sentiment_score,
            "intent": self.primary_intent.value,
            "intent_confidence": self.confidence,
            "conversation_type": self.conversation_type.value,
            "user_state": self.user_state,
            "urgency_level": self.urgency_level,
            "frustration_level": self.frustration_level,
            "formality_preference": self.formality_preference,
            "inferred_age_group": self.inferred_age_group,
            "age_confidence": self.age_confidence,
            "age_indicators": self.age_indicators,
            "technical_level": self.technical_level,
            "situation_type": self.situation_type,
            "key_pain_points": self.key_pain_points,
            "time_sensitivity": self.time_sensitivity,
            "escalation_indicators": self.escalation_indicators,
            "churn_indicators": self.churn_indicators,
            "tokens_used": self.tokens_used,
            "used_ai": self.used_ai
        }

class ContextUnderstandingEngine:
    """
    HYBRID context analysis engine
    - Pattern matching for high-confidence cases (0 tokens)
    - AI fallback for ambiguous cases (uses tokens)
    """
    
    # Confidence threshold for pattern-only mode
    PATTERN_CONFIDENCE_THRESHOLD = 0.85
    
    def __init__(self, client: OpenAI):
        self.client = client
        
        # Pattern libraries for fast-path detection
        self.emotion_patterns = self._build_emotion_patterns()
        self.intent_patterns = self._build_intent_patterns()
        self.age_patterns = self._build_age_patterns()
        self.risk_patterns = self._build_risk_patterns()
    
    def _build_emotion_patterns(self) -> Dict:
        """Emotion detection patterns for CloudBridge contexts"""
        return {
            "concerned": ["concerned", "worried", "unsure", "nervous", "cautious", "hesitant"],
            "frustrated": ["frustrated", "annoying", "keep happening", "third time", "still not working"],
            "confused": ["confused", "don't understand", "unclear", "lost", "how does", "what does"],
            "stressed": ["urgent", "deadline", "critical", "production down", "can't afford"],
            "relieved": ["thank", "appreciate", "that worked", "fixed", "resolved"],
            "disappointed": ["disappointed", "expected more", "thought it would", "not what we needed"],
            "anxious": ["data loss", "security", "breach", "worried about", "what if"],
            "satisfied": ["works great", "exactly what", "perfect", "thank you", "appreciate"],
            "angry": ["unacceptable", "ridiculous", "terrible", "worst", "incompetent"]
        }
    
    def _build_intent_patterns(self) -> Dict:
        """Intent classification patterns"""
        return {
            Intent.PROBLEM_REPORT: {
                "patterns": ["not working", "broken", "error", "bug", "crash", "issue", "problem", "can't", "won't", "doesn't"],
                "confidence": 0.9
            },
            Intent.QUESTION: {
                "patterns": ["how do", "how can", "what is", "where is", "when", "which", "can you", "is it possible"],
                "confidence": 0.85
            },
            Intent.COMPLAINT: {
                "patterns": ["disappointed", "unacceptable", "terrible", "worst", "frustrated with", "waste of time"],
                "confidence": 0.95
            },
            Intent.FEATURE_REQUEST: {
                "patterns": ["would be great if", "could you add", "feature request", "suggestion", "it would be nice"],
                "confidence": 0.9
            },
            Intent.PRAISE: {
                "patterns": ["love", "great", "excellent", "perfect", "amazing", "works well", "exactly what"],
                "confidence": 0.9
            },
            Intent.SECURITY_CONCERN: {
                "patterns": ["security", "breach", "hacked", "unauthorized access", "data leak", "privacy"],
                "confidence": 0.95
            },
            Intent.ESCALATION_REQUEST: {
                "patterns": ["speak to manager", "supervisor", "escalate", "lawyer", "legal"],
                "confidence": 0.98
            },
            Intent.INTEGRATION_HELP: {
                "patterns": ["integration", "api", "webhook", "connect", "sync", "configure"],
                "confidence": 0.85
            },
            Intent.MIGRATION_INQUIRY: {
                "patterns": ["migrate", "migration", "moving from", "switch from", "import from"],
                "confidence": 0.9
            }
        }
    
    def _build_age_patterns(self) -> Dict:
        """Age/career stage patterns"""
        return {
            "early_career": {
                "patterns": ["intern", "junior", "first job", "new to this", "learning", "started recently"],
                "indicators": ["internship", "university", "grad"],
                "formality": "casual",
                "technical": "novice"
            },
            "mid_career": {
                "patterns": ["team lead", "manager", "5 years", "experienced", "been doing this"],
                "indicators": ["managing", "team of", "responsible for"],
                "formality": "neutral",
                "technical": "intermediate"
            },
            "senior": {
                "patterns": ["cto", "vp", "director", "architect", "senior", "decades"],
                "indicators": ["strategic", "enterprise", "organization"],
                "formality": "professional",
                "technical": "expert"
            }
        }
    
    def _build_risk_patterns(self) -> Dict:
        """Risk indicator patterns"""
        return {
            "escalation": ["lawyer", "legal", "sue", "bbb", "report", "attorney", "lawsuit"],
            "churn": ["cancel", "switching to", "moving to", "done with", "waste of money", "competitor"],
            "security_critical": ["breach", "hacked", "unauthorized", "data loss", "compromised", "stolen"]
        }
    
    def analyze(
        self,
        message: str,
        history: Optional[List] = None,
        user_profile: Optional[Dict] = None
    ) -> ContextAnalysis:
        """
        HYBRID analysis: Pattern matching first, AI fallback if needed
        """
        msg_lower = message.lower()
        
        # Step 1: Fast-path pattern matching
        emotion, emotion_intensity, emotion_matched = self._detect_emotion_fast(msg_lower, message)
        age_group, age_conf, age_indicators = self._detect_age_fast(msg_lower, message)
        intent, intent_conf, intent_matched = self._detect_intent_fast(msg_lower, message)
        
        # Step 2: Risk indicators
        escalation_signals = [p for p in self.risk_patterns["escalation"] if p in msg_lower]
        churn_signals = [p for p in self.risk_patterns["churn"] if p in msg_lower]
        security_signals = [p for p in self.risk_patterns.get("security_critical", []) if p in msg_lower]
        
        # Step 3: Decide if patterns are good enough
        use_patterns = (
            intent_conf >= self.PATTERN_CONFIDENCE_THRESHOLD and
            emotion_matched and
            intent_matched
        )
        
        if use_patterns:
            # HIGH CONFIDENCE - Use pattern matching (0 tokens)
            analysis = self._build_analysis_from_patterns(
                message, emotion, emotion_intensity, age_group, age_conf, 
                age_indicators, intent, escalation_signals, churn_signals, security_signals
            )
            analysis.tokens_used = 0
            analysis.used_ai = False
            return analysis
        
        # LOW CONFIDENCE - Use AI analysis
        return self._deep_ai_analysis(
            message, history, user_profile, 
            emotion, age_group, intent, escalation_signals, churn_signals, security_signals
        )
    
    def _detect_emotion_fast(self, msg_lower: str, msg: str) -> Tuple[str, int, bool]:
        """Fast emotion detection - returns (emotion, intensity, matched)"""
        detected_emotions = []
        
        for emotion, patterns in self.emotion_patterns.items():
            if any(p in msg_lower for p in patterns):
                detected_emotions.append(emotion)
        
        if not detected_emotions:
            return "neutral", 3, False  # No match
        
        # Intensity based on punctuation and caps
        intensity = 3
        if "!!!" in msg or "?!!" in msg:
            intensity = 5
        elif "!!" in msg or msg.isupper():
            intensity = 4
        elif "!" in msg:
            intensity = 4
        
        # Return strongest emotion (prioritize negative emotions)
        priority = ["angry", "anxious", "stressed", "frustrated", "disappointed", "concerned", "confused", "satisfied", "relieved"]
        for emo in priority:
            if emo in detected_emotions:
                return emo, intensity, True
        
        return detected_emotions[0], intensity, True
    
    def _detect_age_fast(self, msg_lower: str, msg: str) -> Tuple[str, str, List[str]]:
        """Fast age/career stage detection"""
        for age_group, config in self.age_patterns.items():
            patterns = config.get("patterns", [])
            pattern_matches = [p for p in patterns if p in msg_lower]
            
            if pattern_matches:
                indicators = config["indicators"] + pattern_matches
                return age_group, "high", indicators
        
        return "mid_career", "low", ["no strong indicators"]
    
    def _detect_intent_fast(self, msg_lower: str, msg: str) -> Tuple[Intent, float, bool]:
        """Fast intent detection - returns (intent, confidence, matched)"""
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            patterns = config["patterns"]
            confidence = config["confidence"]
            
            if any(p in msg_lower for p in patterns):
                intent_scores[intent] = confidence
        
        if not intent_scores:
            return Intent.QUESTION, 0.6, False  # Default, low confidence
        
        # Return highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1], True
    
    def _build_analysis_from_patterns(
        self,
        message: str,
        emotion: str,
        emotion_intensity: int,
        age_group: str,
        age_conf: str,
        age_indicators: List[str],
        intent: Intent,
        escalation_signals: List[str],
        churn_signals: List[str],
        security_signals: List[str]
    ) -> ContextAnalysis:
        """Build analysis from pattern matching results"""
        
        msg_lower = message.lower()
        
        # Determine conversation type
        if intent == Intent.SECURITY_CONCERN or security_signals:
            conv_type = ConversationType.SECURITY_ISSUE
        elif intent == Intent.COMPLAINT:
            conv_type = ConversationType.COMPLAINT_RESOLUTION
        elif intent == Intent.PROBLEM_REPORT:
            conv_type = ConversationType.SUPPORT
        elif intent == Intent.ESCALATION_REQUEST:
            conv_type = ConversationType.EMERGENCY
        elif intent == Intent.INTEGRATION_HELP:
            conv_type = ConversationType.TECHNICAL_IMPLEMENTATION
        else:
            conv_type = ConversationType.SUPPORT
        
        # Calculate urgency
        urgency = 3
        if emotion in ["anxious", "stressed"] or "urgent" in msg_lower:
            urgency = 5
        elif intent in [Intent.PROBLEM_REPORT, Intent.SECURITY_CONCERN]:
            urgency = 4
        elif emotion in ["frustrated", "angry"]:
            urgency = 4
        
        # Technical level from age pattern
        age_config = self.age_patterns.get(age_group, {})
        technical_level = age_config.get("technical", "intermediate")
        formality = age_config.get("formality", "neutral")
        
        # Sentiment
        emotion_values = {
            "angry": -0.9, "anxious": -0.6, "stressed": -0.7, "frustrated": -0.7,
            "disappointed": -0.6, "concerned": -0.4, "confused": -0.3, "neutral": 0.0,
            "satisfied": 0.7, "relieved": 0.6
        }
        sentiment = emotion_values.get(emotion, 0.0)
        
        return ContextAnalysis(
            primary_emotion=emotion,
            emotion_intensity=emotion_intensity,
            secondary_emotions=[],
            sentiment_score=sentiment,
            primary_intent=intent,
            confidence=0.85,  # Pattern-based confidence
            multi_intent=False,
            intent_details={},
            conversation_type=conv_type,
            user_state=emotion,
            urgency_level=urgency,
            frustration_level=emotion_intensity if emotion in ["frustrated", "angry"] else 1,
            formality_preference=formality,
            directness=0.7,
            verbosity="moderate",
            inferred_age_group=age_group,
            age_confidence=age_conf,
            age_indicators=age_indicators,
            cultural_signals=[],
            technical_level=technical_level,
            domain_knowledge="intermediate",
            situation_type=self._determine_situation(intent, emotion),
            key_pain_points=self._extract_pain_points(msg_lower),
            mentioned_features=[],
            time_sensitivity=urgency >= 4,
            escalation_indicators=escalation_signals,
            churn_indicators=churn_signals,
            satisfaction_indicators=[],
            tokens_used=0,
            used_ai=False
        )
    
    def _deep_ai_analysis(
        self,
        message: str,
        history: Optional[List],
        user_profile: Optional[Dict],
        emotion_hint: str,
        age_hint: str,
        intent_hint: Intent,
        escalation_signals: List[str],
        churn_signals: List[str],
        security_signals: List[str]
    ) -> ContextAnalysis:
        """AI-powered deep analysis when patterns are insufficient"""
        
        # Build prompt
        system_prompt = f"""Analyze this CloudBridge support message comprehensively.

Pattern hints (may be inaccurate):
- Emotion hint: {emotion_hint}
- Age hint: {age_hint}
- Intent hint: {intent_hint.value}

Return JSON with:
{{
  "emotion": "concerned|frustrated|confused|stressed|relieved|disappointed|anxious|satisfied|angry|neutral",
  "emotion_intensity": 1-5,
  "intent": "problem_report|question|complaint|feature_request|praise|clarification|churn_signal|small_talk|closing|escalation_request|integration_help|security_concern|migration_inquiry",
  "confidence": 0.0-1.0,
  "conversation_type": "support|sales|onboarding|feedback|complaint_resolution|chitchat|emergency|technical_implementation|security_issue",
  "urgency_level": 1-5,
  "technical_level": "novice|intermediate|expert",
  "formality_preference": "casual|neutral|formal",
  "age_group": "early_career|mid_career|senior",
  "situation_type": "string",
  "key_pain_points": ["list"],
  "time_sensitivity": true|false
}}

ONLY return valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            # Parse AI response
            ai_text = response.choices[0].message.content.strip()
            
            if ai_text.startswith('```'):
                ai_text = ai_text.split('```')[1]
                if ai_text.startswith('json'):
                    ai_text = ai_text[4:]
                ai_text = ai_text.strip()
            
            import json
            ai_data = json.loads(ai_text)
            
            # Build analysis from AI data
            analysis = self._build_analysis_from_ai(message, ai_data, escalation_signals, churn_signals, security_signals)
            analysis.tokens_used = tokens_used
            analysis.used_ai = True
            
            return analysis
            
        except Exception as e:
            print(f"AI analysis failed: {e}, using pattern fallback")
            # Emergency fallback
            analysis = self._build_analysis_from_patterns(
                message, emotion_hint, 3, age_hint, "low", [],
                intent_hint, escalation_signals, churn_signals, security_signals
            )
            analysis.tokens_used = 0
            analysis.used_ai = False
            return analysis
    
    def _build_analysis_from_ai(
        self,
        message: str,
        ai_data: Dict,
        escalation_signals: List[str],
        churn_signals: List[str],
        security_signals: List[str]
    ) -> ContextAnalysis:
        """Build ContextAnalysis from AI JSON response"""
        
        emotion = ai_data.get('emotion', 'neutral')
        intent_str = ai_data.get('intent', 'question')
        
        # Map intent string to Intent enum
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.QUESTION
        
        # Map conversation type
        try:
            conv_type = ConversationType(ai_data.get('conversation_type', 'support'))
        except ValueError:
            conv_type = ConversationType.SUPPORT
        
        # Sentiment from emotion
        emotion_values = {
            "angry": -0.9, "anxious": -0.6, "stressed": -0.7, "frustrated": -0.7,
            "disappointed": -0.6, "concerned": -0.4, "confused": -0.3, "neutral": 0.0,
            "satisfied": 0.7, "relieved": 0.6
        }
        sentiment = emotion_values.get(emotion, 0.0)
        
        return ContextAnalysis(
            primary_emotion=emotion,
            emotion_intensity=ai_data.get('emotion_intensity', 3),
            secondary_emotions=[],
            sentiment_score=sentiment,
            primary_intent=intent,
            confidence=ai_data.get('confidence', 0.8),
            multi_intent=False,
            intent_details={},
            conversation_type=conv_type,
            user_state=emotion,
            urgency_level=ai_data.get('urgency_level', 3),
            frustration_level=ai_data.get('emotion_intensity', 3) if emotion in ["frustrated", "angry"] else 1,
            formality_preference=ai_data.get('formality_preference', 'neutral'),
            directness=0.7,
            verbosity="moderate",
            inferred_age_group=ai_data.get('age_group', 'mid_career'),
            age_confidence="medium",
            age_indicators=[],
            cultural_signals=[],
            technical_level=ai_data.get('technical_level', 'intermediate'),
            domain_knowledge="intermediate",
            situation_type=ai_data.get('situation_type', 'general_inquiry'),
            key_pain_points=ai_data.get('key_pain_points', []),
            mentioned_features=[],
            time_sensitivity=ai_data.get('time_sensitivity', False),
            escalation_indicators=escalation_signals,
            churn_indicators=churn_signals,
            satisfaction_indicators=[],
            tokens_used=0,  
            used_ai=True    
        )
    
    def _determine_situation(self, intent: Intent, emotion: str) -> str:
        """Determine situation type"""
        if intent == Intent.SECURITY_CONCERN:
            return "security_incident"
        elif intent == Intent.PROBLEM_REPORT:
            return "technical_issue"
        elif intent == Intent.COMPLAINT:
            return "customer_dissatisfaction"
        elif intent == Intent.MIGRATION_INQUIRY:
            return "migration_planning"
        elif intent == Intent.INTEGRATION_HELP:
            return "integration_setup"
        else:
            return "general_inquiry"
    
    def _extract_pain_points(self, msg_lower: str) -> List[str]:
        """Extract key pain points"""
        pain_points = []
        
        if any(w in msg_lower for w in ["slow", "performance", "lag"]):
            pain_points.append("performance")
        if any(w in msg_lower for w in ["sync", "not syncing", "synchronization"]):
            pain_points.append("synchronization")
        if any(w in msg_lower for w in ["security", "breach", "unauthorized"]):
            pain_points.append("security")
        if any(w in msg_lower for w in ["billing", "charge", "payment"]):
            pain_points.append("billing")
        
        return pain_points