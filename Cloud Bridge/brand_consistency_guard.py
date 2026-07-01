"""
Brand Consistency Guard - Enforces brand voice alignment across all responses
Validates linguistic DNA, forbidden patterns, and brand value adherence
CloudBridge Technology Solutions Edition
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re
from openai import OpenAI

@dataclass
class BrandViolation:
    """Represents a brand consistency violation"""
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "forbidden_phrase", "tone_mismatch", "value_conflict", etc.
    description: str
    location: str  
    suggestion: Optional[str] = None

@dataclass
class BrandValidationResult:
    """Result of brand consistency validation"""
    overall_score: float  # 0-1, where 1 is perfect brand alignment
    passed: bool
    violations: List[BrandViolation]
    scores: Dict[str, float]
    detailed_feedback: str

class BrandConsistencyGuard:
    """
    Enforces brand consistency across all generated responses
    - Validates linguistic DNA (vocabulary, phrasing patterns)
    - Detects forbidden language (corporate buzzwords, off-brand phrases)
    - Ensures value alignment
    - Checks tone boundaries
    """
    
    def __init__(self, brand_config: Dict, client: OpenAI, threshold: float = 0.75):
        """
        Args:
            brand_config: Brand configuration (BRAND_ETHOS)
            client: OpenAI client for semantic validation
            threshold: Minimum score to pass validation (0-1)
        """
        self.brand_config = brand_config
        self.client = client
        self.threshold = threshold
        
        # Extract brand attributes
        self.brand_name = brand_config.get("name", "CloudBridge")
        self.core_values = brand_config.get("core_values", [])
        self.personality = brand_config.get("personality", "")
        self.voice_guidelines = brand_config.get("voice_guidelines", {})
        
        # Build forbidden patterns
        self.forbidden_phrases = self._build_forbidden_patterns()
        self.required_patterns = self._build_required_patterns()
        
        # Linguistic DNA fingerprint
        self.linguistic_dna = self._build_linguistic_dna()
        
    def _build_forbidden_patterns(self) -> Dict[str, List[str]]:
        """Build patterns that violate brand voice"""
        forbidden = {
            "corporate_buzzwords": [
                r"\bsynerg(y|ize|istic)\b", r"\bleverag(e|ing)\b", r"\becosystem\b",
                r"\bparadigm\b", r"\bdisrupt(ive)?\b", r"\bgame[ -]changer\b",
                r"\bcircle back\b", r"\btouch base\b", r"\blow[- ]hanging fruit\b",
                r"\bmove the needle\b", r"\bdrink the kool-aid\b",
                r"\bthought leader(ship)?\b", r"\bbest[- ]in[- ]class\b",
                r"\bcutting[- ]edge\b", r"\brevolutionary breakthrough\b",
                r"\bnext[- ]generation\b", r"\bworld[- ]class\b"
            ],
            "hype_language": [
                r"\bamazing\b", r"\bincredible\b", r"\bunbelievable\b",
                r"\bspectacular\b", r"\bphenomenal\b", r"\bfantastic\b",
                r"\btransform your (business|life|work)\b",
                r"\bchanges? everything\b", r"\bexperience the future\b",
                r"\bunprecedented opportunity\b"
            ],
            "pressure_tactics": [
                r"\bact now\b", r"\blimited time\b", r"\bdon'?t miss out\b",
                r"\bwhile supplies last\b", r"\bonce in a lifetime\b",
                r"\beveryone'?s (switching|using|doing)\b",
                r"\bshouldn'?t you\b"
            ],
            "condescending": [
                r"\beven you can\b", r"\bsimple enough for anyone\b",
                r"\bdon'?t worry,? it'?s easy\b", r"\bjust|simply|merely (click|do|follow)\b",
                r"\bit'?s not (that )?hard\b", r"\bobviously|clearly (this|you)\b",
                r"\bas (I|we) (already )?said\b"
            ],
            "overly_formal": [
                r"we would like to inform you",
                r"please be advised",
                r"at your earliest convenience",
                r"pursuant to",
                r"henceforth",
                r"aforementioned",
                r"herein|thereof|whereby"
            ],
            "generic_template": [
                r"I can understand (how|that)",
                r"let's see (how|if) we can",
                r"if you'?re still (facing|having|experiencing)",
                r"might help to",
                r"you may (want to|need to)",
                r"feel free to",
                r"dear valued customer",
                r"thank you for (contacting|reaching out to) us",
                r"is there anything else (I|we) can (help|assist) you with",
                r"please don'?t hesitate to",
                r"we apologize for any inconvenience",
                r"\[.*?\]"  # Placeholder brackets
            ],
            "vague_promises": [
                r"\btransform(s)? your\b",
                r"\bexperience the\b",
                r"\bdiscover your potential\b",
                r"\bunlock your\b",
                r"\bthe future of\b"
            ]
        }
        
        # Add brand-specific forbidden phrases from config
        brand_forbidden = self.voice_guidelines.get("dont", [])
        forbidden["brand_specific"] = [
            re.escape(phrase.lower()) for phrase in brand_forbidden
        ]
        
        return forbidden
    
    def _build_required_patterns(self) -> Dict[str, List[str]]:
        """Build patterns that should appear in brand voice"""
        required = {
            "contractions": [r"\b(I'm|we're|you're|it's|that's|don't|can't|won't|here's|there's)\b"],
            "clarity_markers": [
                r"\b(clear|simple|straightforward|accessible|easy to understand)\b",
                r"\b(here'?s (how|what|why))\b"
            ],
            "outcome_focus": [
                r"\b(so you can|which means|this lets you|enabling you to)\b",
                r"\b(accomplish|achieve|get done|complete)\b"
            ],
            "inclusive_language": [
                r"\b(everyone|every team|all|anyone|your way)\b"
            ]
        }
        
        return required
    
    def _build_linguistic_dna(self) -> Dict:
        """Build linguistic fingerprint of CloudBridge brand voice"""
        return {
            "avg_sentence_length": (8, 18),    # Shorter, clearer sentences
            "contraction_ratio": (0.2, 0.5),   # Moderate contractions for professionalism
            "avg_word_length": (4, 5.5),       # Simple, accessible vocabulary
            "formality_score": (0.4, 0.7),     # Professional but approachable
            "clarity_density": (0.3, 0.6),     # High focus on clarity
            "question_ratio": (0.0, 0.2),      # Fewer questions, more statements
            "jargon_tolerance": 0.05,          # Very low technical jargon
        }
    
    def validate(
        self, 
        response: str, 
        context: Optional[Dict] = None,
        conversation_type: Optional[str] = None
    ) -> BrandValidationResult:
        """
        Comprehensive brand validation
        
        Args:
            response: Generated response to validate
            context: Optional context (emotion, urgency, etc.)
            conversation_type: Type of conversation (support, sales, etc.)
        
        Returns:
            BrandValidationResult with scores and violations
        """
        violations = []
        scores = {}
        
        # 1. Forbidden phrase detection
        forbidden_score, forbidden_violations = self._check_forbidden_phrases(response)
        violations.extend(forbidden_violations)
        scores["forbidden_phrase_check"] = forbidden_score
        
        # 2. Linguistic DNA analysis
        dna_score, dna_violations = self._check_linguistic_dna(response)
        violations.extend(dna_violations)
        scores["linguistic_dna"] = dna_score
        
        # 3. Tone boundary validation
        tone_score, tone_violations = self._check_tone_boundaries(response, context)
        violations.extend(tone_violations)
        scores["tone_boundaries"] = tone_score
        
        # 4. Value alignment check
        value_score, value_violations = self._check_value_alignment(response)
        violations.extend(value_violations)
        scores["value_alignment"] = value_score
        
        # 5. Clarity and accessibility check
        clarity_score, clarity_violations = self._check_clarity(response)
        violations.extend(clarity_violations)
        scores["clarity"] = clarity_score
        
        # 6. Semantic brand alignment (AI-powered)
        semantic_score = self._check_semantic_alignment(response, context)
        scores["semantic_alignment"] = semantic_score
        
        # Calculate overall score (weighted average)
        weights = {
            "forbidden_phrase_check": 2.5,
            "linguistic_dna": 1.5,
            "tone_boundaries": 2.0,
            "value_alignment": 2.5,
            "clarity": 2.0,
            "semantic_alignment": 1.5
        }
        
        overall_score = sum(
            scores[k] * weights[k] for k in scores
        ) / sum(weights.values())
        
        passed = overall_score >= self.threshold and not any(
            v.severity == "critical" for v in violations
        )
        
        # Generate detailed feedback
        detailed_feedback = self._generate_feedback(scores, violations, overall_score)
        
        return BrandValidationResult(
            overall_score=overall_score,
            passed=passed,
            violations=violations,
            scores=scores,
            detailed_feedback=detailed_feedback
        )
    
    def _check_forbidden_phrases(self, response: str) -> tuple[float, List[BrandViolation]]:
        """Check for forbidden phrases and patterns"""
        violations = []
        response_lower = response.lower()
        
        total_checks = 0
        failed_checks = 0
        
        for category, patterns in self.forbidden_phrases.items():
            for pattern in patterns:
                total_checks += 1
                matches = list(re.finditer(pattern, response_lower, re.IGNORECASE))
                
                if matches:
                    failed_checks += 1
                    severity_map = {
                        "corporate_buzzwords": "critical",
                        "hype_language": "high",
                        "pressure_tactics": "critical",
                        "condescending": "critical",
                        "vague_promises": "high"
                    }
                    severity = severity_map.get(category, "medium")
                    
                    for match in matches:
                        violations.append(BrandViolation(
                            severity=severity,
                            category=f"forbidden_{category}",
                            description=f"Found forbidden phrase: '{match.group()}'",
                            location=f"Position {match.start()}-{match.end()}",
                            suggestion=self._suggest_alternative(match.group(), category)
                        ))
        
        score = 1.0 - (failed_checks / max(total_checks, 1))
        return score, violations
    
    def _check_linguistic_dna(self, response: str) -> tuple[float, List[BrandViolation]]:
        """Check if response matches CloudBridge's linguistic fingerprint"""
        violations = []
        scores = []
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5, violations
        
        # Sentence length
        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
        min_len, max_len = self.linguistic_dna["avg_sentence_length"]
        if min_len <= avg_sent_len <= max_len:
            scores.append(1.0)
        else:
            score = max(0, 1 - abs(avg_sent_len - (min_len + max_len) / 2) / max_len)
            scores.append(score)
            if score < 0.6:
                violations.append(BrandViolation(
                    severity="low",
                    category="linguistic_dna",
                    description=f"Average sentence length ({avg_sent_len:.1f} words) outside brand range ({min_len}-{max_len})",
                    location="Overall structure",
                    suggestion=f"CloudBridge uses shorter, clearer sentences. Aim for {(min_len + max_len) // 2} words per sentence"
                ))
        
        # Contraction ratio
        contractions = len(re.findall(r"\b(I'm|we're|you're|it's|that's|don't|can't|won't|haven't|hasn't|here's|there's)\b", response, re.IGNORECASE))
        contraction_ratio = contractions / len(sentences)
        min_ratio, max_ratio = self.linguistic_dna["contraction_ratio"]
        if min_ratio <= contraction_ratio <= max_ratio:
            scores.append(1.0)
        else:
            score = max(0, 1 - abs(contraction_ratio - (min_ratio + max_ratio) / 2) / max_ratio)
            scores.append(score)
            if score < 0.6:
                violations.append(BrandViolation(
                    severity="medium",
                    category="linguistic_dna",
                    description=f"Contraction usage ({contraction_ratio:.1%}) outside brand range ({min_ratio:.0%}-{max_ratio:.0%})",
                    location="Overall tone",
                    suggestion="Use moderate contractions for professional yet approachable tone" if contraction_ratio < min_ratio else "Slightly reduce contractions"
                ))
        
        overall_score = sum(scores) / len(scores) if scores else 0.5
        return overall_score, violations
    
    def _check_clarity(self, response: str) -> tuple[float, List[BrandViolation]]:
        """Check for clarity and jargon-free communication"""
        violations = []
        response_lower = response.lower()
        
        # Technical jargon check
        tech_jargon = [
            r"\bAPI\b", r"\bSDK\b", r"\bSaaS\b", r"\bIaaS\b", r"\bPaaS\b",
            r"\bmicroservices\b", r"\bkubernetes\b", r"\bcontainerization\b",
            r"\bserverless\b", r"\bCI/CD\b", r"\bDevOps\b"
        ]
        
        jargon_count = sum(1 for pattern in tech_jargon if re.search(pattern, response, re.IGNORECASE))
        total_words = len(response.split())
        jargon_ratio = jargon_count / max(total_words, 1)
        
        if jargon_ratio > self.linguistic_dna["jargon_tolerance"]:
            violations.append(BrandViolation(
                severity="medium",
                category="clarity",
                description=f"Too much technical jargon ({jargon_count} terms found)",
                location="Word choice",
                suggestion="Explain technical concepts in plain language or provide context"
            ))
        
        # Complex sentence structures
        complex_patterns = [
            r"\b(notwithstanding|aforementioned|heretofore|whereby)\b",
            r"(which|that|who)\s+\w+\s+(which|that|who)",  # Nested clauses
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, response_lower):
                violations.append(BrandViolation(
                    severity="low",
                    category="clarity",
                    description="Complex sentence structure detected",
                    location="Sentence structure",
                    suggestion="Break into simpler, clearer sentences"
                ))
        
        score = 1.0 if not violations else max(0, 1.0 - len(violations) * 0.2)
        return score, violations
    
    def _check_tone_boundaries(self, response: str, context: Optional[Dict]) -> tuple[float, List[BrandViolation]]:
        """Validate tone stays within CloudBridge boundaries"""
        violations = []
        
        # Check for inappropriate casualness in serious security contexts
        if context and context.get("topic") in ["security", "data_breach", "privacy"]:
            casual_markers = ["no worries", "no biggie", "all good", "cool", "awesome"]
            if any(marker in response.lower() for marker in casual_markers):
                violations.append(BrandViolation(
                    severity="critical",
                    category="tone_boundary",
                    description="Overly casual tone in serious security context",
                    location="Overall tone",
                    suggestion="Maintain professional, reassuring tone for security matters"
                ))
        
        # Check for hype or exaggeration
        hype_markers = ["amazing", "incredible", "unbelievable", "spectacular", "mind-blowing"]
        found_hype = [m for m in hype_markers if m in response.lower()]
        if found_hype:
            violations.append(BrandViolation(
                severity="high",
                category="tone_boundary",
                description=f"Hype language detected: {', '.join(found_hype)}",
                location="Word choice",
                suggestion="Use measured, professional language instead"
            ))
        
        # Check for excessive formality
        formal_markers = ["pursuant to", "herein", "aforementioned", "heretofore"]
        found_formal = [m for m in formal_markers if m in response.lower()]
        if found_formal:
            violations.append(BrandViolation(
                severity="medium",
                category="tone_boundary",
                description=f"Overly formal language: {', '.join(found_formal)}",
                location="Word choice",
                suggestion="Use clear, accessible language"
            ))
        
        score = 1.0 if not violations else max(0, 1.0 - len(violations) * 0.25)
        return score, violations
    
    def _check_value_alignment(self, response: str) -> tuple[float, List[BrandViolation]]:
        """Check if response aligns with CloudBridge values"""
        violations = []
        response_lower = response.lower()
        
        # Check for inclusivity
        exclusive_patterns = [
            r"\bfor (experts|professionals|advanced users) only\b",
            r"\bnot for (beginners|novices)\b",
            r"\byou need (to be|to know)\b"
        ]
        
        for pattern in exclusive_patterns:
            if re.search(pattern, response_lower):
                violations.append(BrandViolation(
                    severity="high",
                    category="value_alignment",
                    description="Language conflicts with 'Inclusivity' value",
                    location="Accessibility",
                    suggestion="CloudBridge is for everyone - use inclusive language"
                ))
        
        # Check for transparency
        evasive_patterns = [
            r"we (can't|cannot) disclose",
            r"that'?s (proprietary|confidential)",
            r"we'?re not at liberty",
            r"I can'?t (really )?say"
        ]
        
        for pattern in evasive_patterns:
            if re.search(pattern, response_lower):
                violations.append(BrandViolation(
                    severity="medium",
                    category="value_alignment",
                    description="Evasive language conflicts with 'Trust' value",
                    location="Transparency",
                    suggestion="Be direct about what you can and cannot do, and why"
                ))
        
        # Check for helpfulness (avoiding deflection)
        deflection_patterns = [
            r"that'?s not my (job|responsibility|department)",
            r"you'?ll need to (contact|reach out to)",
            r"I can'?t help with that"
        ]
        
        for pattern in deflection_patterns:
            if re.search(pattern, response_lower):
                violations.append(BrandViolation(
                    severity="high",
                    category="value_alignment",
                    description="Deflection conflicts with 'Collaboration' value",
                    location="Helpfulness",
                    suggestion="Offer to help connect them or provide alternative solutions"
                ))
        
        score = 1.0 if not violations else max(0, 1.0 - len(violations) * 0.25)
        return score, violations
    
    def _check_semantic_alignment(self, response: str, context: Optional[Dict]) -> float:
        """AI-powered semantic brand alignment check"""
        try:
            prompt = f"""Analyze if this response sounds like it's from {self.brand_name}.

Brand Personality: {self.personality}
Core Values: {', '.join(self.core_values)}

CloudBridge is a professional technology company that:
- Uses clear, jargon-free language
- Focuses on outcomes over features
- Is empowering but never patronizing
- Is professional but approachable
- Enables rather than prescribes

Response to analyze:
"{response}"

Does this response authentically embody the CloudBridge brand voice and values?

Return JSON:
{{
    "brand_alignment_score": 0.0-1.0,
    "sounds_like_brand": true/false,
    "reasoning": "brief explanation"
}}"""

            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a brand voice expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = completion.choices[0].message.content
            
            # Parse JSON
            import json
            result_json = json.loads(result)
            return result_json.get("brand_alignment_score", 0.7)
            
        except Exception as e:
            print(f"⚠️ Semantic alignment check failed: {e}")
            return 0.7  # Neutral default
    
    def _suggest_alternative(self, forbidden_phrase: str, category: str) -> str:
        """Suggest alternative phrasing for CloudBridge voice"""
        alternatives = {
            "synergy": "working together",
            "leverage": "use",
            "ecosystem": "platform / environment",
            "paradigm": "model / approach",
            "disrupt": "improve / change",
            "circle back": "follow up / get back to you",
            "touch base": "connect",
            "amazing": "effective / powerful",
            "incredible": "impressive / strong",
            "revolutionary": "new / improved",
            "transform your business": "help your business achieve more",
            "act now": "get started",
            "limited time": "(remove urgency)",
            "thank you for contacting us": "Thanks for reaching out",
            "is there anything else": "(end naturally or ask specific follow-up)",
            "please don't hesitate": "Feel free to reach out",
            "even you can": "easy to use / accessible"
        }
        
        phrase_lower = forbidden_phrase.lower()
        for key, alt in alternatives.items():
            if key in phrase_lower:
                return f"Try: '{alt}'"
        
        return "Rephrase in clear, accessible language"
    
    def _generate_feedback(
        self, 
        scores: Dict[str, float], 
        violations: List[BrandViolation],
        overall_score: float
    ) -> str:
        """Generate detailed feedback report"""
        feedback_parts = [
            f"Overall Brand Alignment: {overall_score:.1%}\n"
        ]
        
        # Score breakdown
        feedback_parts.append("Score Breakdown:")
        for category, score in scores.items():
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            feedback_parts.append(f"  {status} {category.replace('_', ' ').title()}: {score:.1%}")
        
        # Violations summary
        if violations:
            feedback_parts.append(f"\nViolations Found: {len(violations)}")
            
            critical = [v for v in violations if v.severity == "critical"]
            high = [v for v in violations if v.severity == "high"]
            
            if critical:
                feedback_parts.append(f"  🚨 Critical: {len(critical)}")
            if high:
                feedback_parts.append(f"  ⚠️  High: {len(high)}")
            
            # Top 3 violations
            feedback_parts.append("\nTop Issues:")
            for v in violations[:3]:
                feedback_parts.append(f"  • {v.description}")
                if v.suggestion:
                    feedback_parts.append(f"    → {v.suggestion}")
        else:
            feedback_parts.append("\n✅ No violations detected!")
        
        return "\n".join(feedback_parts)
    
    def get_boundaries(self) -> Dict:
        """Get CloudBridge brand tone boundaries for tone adapter"""
        return {
            "max_formality": 0.7,      # Professional but not stiff
            "min_formality": 0.4,      # Not too casual
            "min_clarity": 0.7,        # Always prioritize clarity
            "max_jargon": 0.05,        # Very low jargon tolerance
            "min_directness": 0.6,     # Be direct and clear
            "max_hype": 0.2,           # Minimal hype language
            "forbidden_emotions": ["hype", "pressure", "condescension"],
            "required_emotions": ["helpfulness", "clarity", "empowerment"],
            "linguistic_dna": self.linguistic_dna
        }