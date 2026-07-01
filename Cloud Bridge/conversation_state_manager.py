"""
Conversation State Manager - Tracks context across multi-turn conversations
Monitors sentiment patterns, topic flow, user preferences, and issue resolution
CloudBridge Technology Solutions Edition
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

class ConversationPhase(Enum):
    """Where we are in the conversation journey"""
    GREETING = "greeting"
    PROBLEM_IDENTIFICATION = "problem_identification"
    SOLUTION_EXPLORATION = "solution_exploration"
    RESOLUTION = "resolution"
    FOLLOW_UP = "follow_up"
    CLOSING = "closing"

class SentimentTrend(Enum):
    """How sentiment changes over the conversation"""
    IMPROVING = "improving"           # Getting more positive
    DETERIORATING = "deteriorating"   # Getting more negative
    STABLE_POSITIVE = "stable_positive"
    STABLE_NEGATIVE = "stable_negative"
    STABLE_NEUTRAL = "stable_neutral"
    VOLATILE = "volatile"             # Fluctuating significantly

@dataclass
class ConversationTurn:
    """A single back-and-forth exchange"""
    timestamp: datetime
    user_message: str
    assistant_response: str
    user_emotion: str
    emotion_intensity: int           # 1-5 scale
    intent: str
    topic: str
    sentiment_score: float           # -1 (negative) to 1 (positive)
    urgency_level: int               # 1-5 scale
    tone_used: Dict
    context_snapshot: Dict

@dataclass
class UserProfile:
    """
    What we learn about the user over time
    Helps us adapt our responses to fit their needs
    """
    user_id: Optional[str] = None
    inferred_age_group: Optional[str] = None
    age_confidence: str = "low"                    # How certain we are
    formality_preference: str = "neutral"          # casual/neutral/formal
    communication_style: str = "neutral"           # direct/verbose/casual
    technical_level: str = "intermediate"          # novice/intermediate/expert
    cultural_indicators: List[str] = field(default_factory=list)
    preferred_response_length: str = "moderate"    # brief/moderate/detailed
    emotional_baseline: str = "neutral"
    patience_level: int = 3                        # 1-5 scale
    relationship_stage: str = "first_interaction"  # first/regular/loyal/at_risk
    satisfaction_score: float = 0.5                # 0-1, tracks satisfaction trend
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    def update_from_turn(self, turn: ConversationTurn):
        """Learn from each conversation turn"""
        self.total_interactions += 1
        
        # Track positive vs negative experiences
        if turn.sentiment_score > 0.3:
            self.positive_interactions += 1
        elif turn.sentiment_score < -0.3:
            self.negative_interactions += 1
        
        # Update relationship stage based on interaction patterns
        if self.total_interactions == 1:
            self.relationship_stage = "first_interaction"
        elif self.total_interactions < 5:
            self.relationship_stage = "regular"
        elif self.positive_interactions / self.total_interactions > 0.7:
            self.relationship_stage = "loyal"
        elif self.negative_interactions / self.total_interactions > 0.5:
            self.relationship_stage = "at_risk"
        
        # Calculate overall satisfaction trend
        if self.total_interactions > 0:
            self.satisfaction_score = self.positive_interactions / self.total_interactions

@dataclass
class Issue:
    """
    Represents something the user needs help with
    Tracks from first mention through resolution
    """
    id: str
    description: str
    category: str              # "technical", "billing", "feature_request", etc.
    severity: str              # "low", "medium", "high", "critical"
    status: str                # "open", "in_progress", "resolved", "needs_escalation"
    first_mentioned: datetime
    last_mentioned: Optional[datetime] = None
    resolution_attempts: int = 0
    resolved: bool = False
    resolution_summary: Optional[str] = None

class ConversationStateManager:
    """
    Keeps track of everything happening in a conversation
    
    What it does:
    - Remembers what's been discussed
    - Learns user preferences over time
    - Tracks how sentiment changes
    - Monitors topics and issues
    - Identifies the conversation phase
    
    Why it matters:
    Conversations feel natural when we remember context. This ensures
    we don't ask the same questions twice and we adapt to how each
    person prefers to communicate.
    """
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Core conversation data
        self.turns: List[ConversationTurn] = []
        self.user_profile = UserProfile(user_id=user_id)
        self.current_phase = ConversationPhase.GREETING
        
        # Topic tracking - what we're discussing
        self.topic_stack: List[str] = []                    
        self.topic_history: List[Tuple[str, datetime]] = [] 
        
        # Issue tracking - what needs solving
        self.active_issues: List[Issue] = []
        self.resolved_issues: List[Issue] = []
        
        # Sentiment tracking - how the user feels over time
        self.sentiment_history: List[Tuple[datetime, float]] = []
        self.sentiment_trend = SentimentTrend.STABLE_NEUTRAL
        
        # Indicators that need attention
        self.needs_escalation: float = 0.0       # 0-1 scale
        self.satisfaction_declining: float = 0.0  # 0-1 scale
        self.repeated_frustration_count: int = 0
        
    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        context: Dict,
        tone: Dict
    ):
        """
        Add a new exchange to the conversation
        Updates all tracking systems automatically
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_response=assistant_response,
            user_emotion=context.get("emotion", "neutral"),
            emotion_intensity=context.get("emotion_intensity", 3),
            intent=context.get("intent", "unknown"),
            topic=context.get("topic", "general"),
            sentiment_score=self._calculate_sentiment_score(context),
            urgency_level=context.get("urgency_level", 2),
            tone_used=tone,
            context_snapshot=context.copy()
        )
        
        self.turns.append(turn)
        self.last_updated = datetime.now()
        
        # Update all tracking systems with new information
        self._update_sentiment_tracking(turn)
        self._update_topic_tracking(turn)
        self._update_user_profile(turn)
        self._update_phase(turn)
        self._update_attention_indicators(turn)
        
    def _calculate_sentiment_score(self, context: Dict) -> float:
        """
        Convert emotion to a sentiment score
        Returns: -1 (very negative) to 1 (very positive)
        """
        emotion_scores = {
            "angry": -0.9,
            "frustrated": -0.7,
            "anxious": -0.5,
            "confused": -0.3,
            "disappointed": -0.6,
            "concerned": -0.4,
            "neutral": 0.0,
            "satisfied": 0.6,
            "grateful": 0.8,
            "excited": 0.9,
            "happy": 0.7,
            "relieved": 0.7
        }
        
        emotion = context.get("emotion", "neutral").lower()
        base_score = emotion_scores.get(emotion, 0.0)
        
        # Adjust based on intensity (3 is neutral baseline)
        intensity = context.get("emotion_intensity", 3)
        intensity_multiplier = intensity / 3.0
        
        return base_score * intensity_multiplier
    
    def _update_sentiment_tracking(self, turn: ConversationTurn):
        """Track how sentiment changes over the conversation"""
        self.sentiment_history.append((turn.timestamp, turn.sentiment_score))
        
        # Keep last 10 turns for trend analysis
        if len(self.sentiment_history) > 10:
            self.sentiment_history = self.sentiment_history[-10:]
        
        # Analyze the trend if we have enough data
        if len(self.sentiment_history) >= 3:
            recent_scores = [score for _, score in self.sentiment_history[-3:]]
            
            # Check if consistently improving or declining
            if all(recent_scores[i] < recent_scores[i+1] for i in range(len(recent_scores)-1)):
                self.sentiment_trend = SentimentTrend.IMPROVING
            elif all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                self.sentiment_trend = SentimentTrend.DETERIORATING
            else:
                # Check overall level if not trending
                avg_score = sum(recent_scores) / len(recent_scores)
                if avg_score > 0.4:
                    self.sentiment_trend = SentimentTrend.STABLE_POSITIVE
                elif avg_score < -0.4:
                    self.sentiment_trend = SentimentTrend.STABLE_NEGATIVE
                else:
                    self.sentiment_trend = SentimentTrend.STABLE_NEUTRAL
    
    def _update_topic_tracking(self, turn: ConversationTurn):
        """Keep track of what's being discussed"""
        topic = turn.topic
        
        self.topic_history.append((topic, turn.timestamp))
        
        # Update active topics stack
        if topic not in self.topic_stack:
            self.topic_stack.append(topic)
        
        # Keep stack focused on recent topics (max 3)
        if len(self.topic_stack) > 3:
            self.topic_stack.pop(0)
    
    def _update_user_profile(self, turn: ConversationTurn):
        """Learn and remember user preferences"""
        self.user_profile.update_from_turn(turn)
        
        # Update age group inference if we detected it
        if turn.context_snapshot.get("inferred_age_group"):
            # Only update if we're more confident now
            if (self.user_profile.age_confidence == "low" or 
                turn.context_snapshot.get("age_confidence") == "high"):
                self.user_profile.inferred_age_group = turn.context_snapshot["inferred_age_group"]
                self.user_profile.age_confidence = turn.context_snapshot.get("age_confidence", "medium")
        
        # Learn formality preference
        if turn.context_snapshot.get("formality_preference"):
            self.user_profile.formality_preference = turn.context_snapshot["formality_preference"]
        
        # Detect technical expertise level from conversation
        msg_lower = turn.user_message.lower()
        technical_terms = ["api", "integration", "sdk", "configuration", "authentication", "deployment"]
        advanced_terms = ["architecture", "infrastructure", "kubernetes", "microservices"]
        
        if any(term in msg_lower for term in advanced_terms):
            self.user_profile.technical_level = "expert"
        elif any(term in msg_lower for term in technical_terms):
            self.user_profile.technical_level = "intermediate"
        elif "confused" in turn.user_emotion or "don't understand" in msg_lower:
            self.user_profile.technical_level = "novice"
    
    def _update_phase(self, turn: ConversationTurn):
        """Identify where we are in the conversation journey"""
        if len(self.turns) == 1:
            self.current_phase = ConversationPhase.GREETING
        elif turn.intent in ["problem_report", "issue", "complaint"]:
            self.current_phase = ConversationPhase.PROBLEM_IDENTIFICATION
        elif turn.intent == "question" and self.current_phase == ConversationPhase.PROBLEM_IDENTIFICATION:
            self.current_phase = ConversationPhase.SOLUTION_EXPLORATION
        elif turn.user_emotion in ["grateful", "satisfied", "relieved"]:
            self.current_phase = ConversationPhase.RESOLUTION
        elif "thank" in turn.user_message.lower() or "bye" in turn.user_message.lower():
            self.current_phase = ConversationPhase.CLOSING
    
    def _update_attention_indicators(self, turn: ConversationTurn):
        """
        Track signals that something needs attention
        (Internal tracking - not shown to users)
        """
        msg_lower = turn.user_message.lower()
        
        # Signals someone may want to speak with management
        escalation_signals = ["manager", "supervisor", "escalate", "unacceptable", "legal"]
        if any(signal in msg_lower for signal in escalation_signals):
            self.needs_escalation = min(1.0, self.needs_escalation + 0.3)
        
        # Signals satisfaction is declining
        declining_signals = ["cancel", "refund", "disappointed", "switching", "competitor", "waste of"]
        if any(signal in msg_lower for signal in declining_signals):
            self.satisfaction_declining = min(1.0, self.satisfaction_declining + 0.25)
        
        # Track repeated frustration
        if turn.user_emotion in ["frustrated", "angry", "disappointed"]:
            self.repeated_frustration_count += 1
            if self.repeated_frustration_count >= 3:
                self.needs_escalation = min(1.0, self.needs_escalation + 0.2)
        else:
            # Reset counter when emotion improves
            self.repeated_frustration_count = max(0, self.repeated_frustration_count - 1)
    
    def add_issue(self, description: str, category: str, severity: str):
        """
        Start tracking a new issue
        Returns the Issue object for reference
        """
        issue = Issue(
            id=f"issue_{len(self.active_issues) + len(self.resolved_issues) + 1}",
            description=description,
            category=category,
            severity=severity,
            status="open",
            first_mentioned=datetime.now()
        )
        self.active_issues.append(issue)
        return issue
    
    def resolve_issue(self, issue_id: str, resolution: str):
        """Mark an issue as resolved"""
        for issue in self.active_issues:
            if issue.id == issue_id:
                issue.resolved = True
                issue.status = "resolved"
                issue.resolution_summary = resolution
                self.resolved_issues.append(issue)
                self.active_issues.remove(issue)
                break
    
    def get_context_summary(self) -> Dict:
        """
        Get a snapshot of current conversation state
        Used to inform the next response
        """
        recent_emotions = [t.user_emotion for t in self.turns[-3:]] if len(self.turns) >= 3 else []
        
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "current_phase": self.current_phase.value,
            "sentiment_trend": self.sentiment_trend.value,
            "current_sentiment": self.sentiment_history[-1][1] if self.sentiment_history else 0.0,
            "recent_emotions": recent_emotions,
            "active_topics": self.topic_stack,
            "user_profile": {
                "age_group": self.user_profile.inferred_age_group,
                "formality": self.user_profile.formality_preference,
                "technical_level": self.user_profile.technical_level,
                "relationship_stage": self.user_profile.relationship_stage,
                "satisfaction_score": self.user_profile.satisfaction_score
            },
            "active_issues_count": len(self.active_issues),
            "resolved_issues_count": len(self.resolved_issues),
            "attention_needed": {
                "escalation_level": self.needs_escalation,
                "satisfaction_declining": self.satisfaction_declining,
                "repeated_frustration": self.repeated_frustration_count
            }
        }
    
    def should_escalate(self) -> bool:
        """
        Determine if this conversation should be escalated to a human
        Returns True when urgent attention is needed
        """
        return (
            self.needs_escalation > 0.7 or
            self.repeated_frustration_count >= 4 or
            any(issue.severity == "critical" for issue in self.active_issues)
        )
    
    def get_conversation_history_text(self, max_turns: int = 5) -> str:
        """
        Format recent conversation as readable text
        Useful for context when generating responses
        """
        recent_turns = self.turns[-max_turns:]
        
        history_parts = []
        for i, turn in enumerate(recent_turns, 1):
            history_parts.append(f"Turn {i}:")
            history_parts.append(f"User: {turn.user_message}")
            history_parts.append(f"Assistant: {turn.assistant_response}")
            history_parts.append("")  
        
        return "\n".join(history_parts)
    
    def to_dict(self) -> Dict:
        """
        Serialize conversation state to dictionary
        Useful for storage, logging, or transfer
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "turn_count": len(self.turns),
            "current_phase": self.current_phase.value,
            "sentiment_trend": self.sentiment_trend.value,
            "user_profile": {
                "age_group": self.user_profile.inferred_age_group,
                "formality": self.user_profile.formality_preference,
                "technical_level": self.user_profile.technical_level,
                "relationship_stage": self.user_profile.relationship_stage,
                "satisfaction_score": self.user_profile.satisfaction_score,
                "total_interactions": self.user_profile.total_interactions
            },
            "active_issues": [
                {
                    "id": issue.id,
                    "description": issue.description,
                    "category": issue.category,
                    "severity": issue.severity,
                    "status": issue.status
                }
                for issue in self.active_issues
            ],
            "attention_indicators": {
                "needs_escalation": self.needs_escalation,
                "satisfaction_declining": self.satisfaction_declining,
                "repeated_frustration_count": self.repeated_frustration_count
            }
        }