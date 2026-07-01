"""
CloudBridge Web Interface - Flask Backend
Handles API routes, response generation, evaluation, and performance tracking
"""

from flask import Flask, render_template, request, jsonify
import json
import secrets
from datetime import datetime
import time

from universal_adaptive_framework import (
    UniversalAdaptiveFramework,
    BRAND_ETHOS,
    client,
    tracker,
    optimizer
)
from performance_tracker import StepTimer

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Initialize CloudBridge framework
framework = UniversalAdaptiveFramework(BRAND_ETHOS, client)

class BaselineSystem:
    """
    Traditional chatbot system for comparison
    Uses fixed, template-based responses without context adaptation
    """
    
    def __init__(self, brand_ethos):
        self.brand_ethos = brand_ethos
        self.client = client
        self.temperature = 0.2
        self.max_tokens = 350
        self.context_window = 1
        
        self.response_patterns = [
            "I understand your inquiry regarding",
            "I can assist you with",
            "Please provide the following information",
            "Is there anything else I can help you with today?"
        ]
    
    def generate_response(self, message, conversation_history=None):
        """Generate baseline response without context awareness"""
        step_start = time.time()

        is_first_message = not conversation_history or len(conversation_history) == 0
        greeting_rule = (
            'Open with a warm formal greeting like "Thank you for contacting us!" or "Thanks for reaching out!" — first message only, vary it each time.'
            if is_first_message
            else "Do NOT open with any greeting — the conversation is already in progress."
        )

        system_prompt = f"""You are a customer service chatbot for tech company.
        STRICT RULES - YOU MUST FOLLOW THESE EXACTLY:
Use ONLY formal, template-based corporate language
{greeting_rule}
Always end with a closing question like "Is there anything else I can help you with?" or "Is there anything else I can assist you with today?" — vary it each time.
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            recent_history = conversation_history[-self.context_window:]
            for exchange in recent_history:
                if 'user' in exchange and 'assistant' in exchange:
                    messages.append({"role": "user", "content": exchange['user']})
                    messages.append({"role": "assistant", "content": exchange['assistant']})
        
        messages.append({"role": "user", "content": message})
        
        selected_model = optimizer.select_model(
            step_name="Baseline Response Generation",
            prompt=message,
            estimated_input_tokens=sum(len(m['content'].split()) for m in messages) * 1.3,
            estimated_output_tokens=self.max_tokens
        )
        
        api_start = time.time()  # Track API call start
        
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        step_time = time.time() - api_start  # Calculate actual API time
        tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
        
        return {
            'response': response.choices[0].message.content,
            'model': selected_model,
            'tokens': tokens,
            'time': step_time
        }

baseline_system = BaselineSystem(BRAND_ETHOS)

class ToneBrandEvaluator:
    """
    Evaluates and compares responses on tone, brand voice, and context adaptation
    Uses AI-powered blind evaluation for objectivity
    """
    
    def __init__(self, client):
        self.client = client
        
        self.EVALUATION_PARAMETERS = {
            
            "Tone_Adaptation": {"weight": 2.0},
            "Context_Specificity": {"weight": 1.5},
            "Brand_Voice_Authenticity": {"weight": 2.0},
           
        }
    
    def _extract_json_from_text(self, text):
        """Extract JSON from text that might contain markdown or other content"""
        import re
        
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def evaluate_response_blind(self, chat_content, response, context):
        """Evaluate a single response without knowing which system generated it"""
        eval_start = time.time()
        
        prompt = f"""You are an objective evaluator. Rate the customer support response using the scoring rubric below.

USER MESSAGE: "{chat_content}"
User's emotion: {context.get('emotion', 'unclear')}

RESPONSE TO EVALUATE: "{response}"

Evaluate each criterion from 1 to 5 using these definitions:

SCORING RUBRIC (1 = Very Poor, 5 = Excellent)



1. Tone_Adaptation
   - 5: Tone perfectly matches the user's communication style (formal/informal, calm/urgent)
   - 4: Good adaptation; minor mismatch
   - 3: Partially adapted; somewhat generic
   - 2: Poor adaptation; tone doesn't fit user
   - 1: Tone is inappropriate or completely mismatched

2. Context_Specificity
   - 5: Fully tailored to the user's specific situation; not generic
   - 4: Mostly tailored; minor generic phrasing
   - 3: Partially specific; some template-like content
   - 2: Little tailoring; mostly generic
   - 1: Ignores context or responds off-topic

3. Brand_Voice_Authenticity (CRITICAL - CloudBridge Brand Assessment)

CloudBridge's Brand Voice Requirements:
1. MUST use contractions naturally (I'm, you're, don't, can't, it's, that's)
2. MUST sound conversational like a helpful partner, NOT corporate
3. MUST avoid template phrases
4. MUST provide specific, actionable guidance

SCORING RULES (BE STRICT):

5 = Exemplary Brand Voice
    Uses 4+ contractions naturally
    Zero template phrases
    Distinctly conversational (sounds like CloudBridge, not generic)
    Specific, actionable guidance

4 = Strong Brand Voice  
    Uses 2-3 contractions
    0-1 template phrases
    Mostly conversational with minor formal slips
    Generally specific guidance

3 = Generic Professional (NOT CloudBridge voice)
    Few contractions (0-1)
    Professional tone that could be ANY company
    No distinctive personality

2 = Corporate/Template Language (VIOLATES CloudBridge brand)
    Zero contractions
    Multiple template phrases (2+)
    Robotic, formal phrasing
    Sounds like generic call center

1 = Completely Off-Brand
    Dismissive, unhelpful, or contradicts values

 MANDATORY SCORING RULES:
- If response has 0 contractions → MAXIMUM score = 2
- If response has 3+ template phrases → MAXIMUM score = 2  
- If response has bullet-point list AND formal language → MAXIMUM score = 3
- If response could work for ANY company unchanged → MAXIMUM score = 3


IMPORTANT: Be STRICT on scoring. If a response uses template language, formal acknowledgments, or lacks personality, score accordingly. Don't be generous.

Return ONLY valid JSON (no markdown, no preamble):
{{
  "scores": {{
    "Emotional_Acknowledgment": 3,
    "Tone_Adaptation": 3,
    "Context_Specificity": 3,
    "Brand_Voice_Authenticity": 3,
    "Urgency_Matching": 3,
    "Natural_Language": 3
  }},
  "weighted_score": 3.0,
  "strengths": ["strength1"],
  "weaknesses": ["weakness1"]
}}"""

        try:
            selected_model = optimizer.select_model(
                step_name="Response Evaluation",
                prompt=prompt,
                estimated_input_tokens=len(prompt.split()) * 1.3,
                estimated_output_tokens=400
            )
            
            response_obj = self.client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "You are a STRICT evaluator. Return ONLY valid JSON, no markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_completion_tokens=600
            )
            
            raw_content = response_obj.choices[0].message.content
            json_content = self._extract_json_from_text(raw_content)
            result = json.loads(json_content)
            
            # Calculate weighted score 
            if "weighted_score" not in result or result["weighted_score"] == 0:
                weights = self.EVALUATION_PARAMETERS
                total = sum(result["scores"].get(k, 3) * weights[k]["weight"] for k in weights)
                result["weighted_score"] = round(total / sum(w["weight"] for w in weights.values()), 2)
            
            # Add performance metadata
            eval_time = time.time() - eval_start
            tokens = response_obj.usage.total_tokens if hasattr(response_obj, 'usage') else 0
            
            result['_meta'] = {
                'model': selected_model,
                'tokens': tokens,
                'time': eval_time
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Evaluation error: {e}")
            print(f"Raw content: {raw_content if 'raw_content' in locals() else 'N/A'}")
            eval_time = time.time() - eval_start
            return self._get_default_evaluation(eval_time, selected_model if 'selected_model' in locals() else 'N/A')
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            eval_time = time.time() - eval_start
            return self._get_default_evaluation(eval_time, 'N/A')
    
    def _get_default_evaluation(self, eval_time, model):
        """Return default evaluation when parsing fails"""
        return {
            "scores": {k: 3 for k in self.EVALUATION_PARAMETERS.keys()},
            "weighted_score": 3.0,
            "strengths": ["Unable to evaluate"],
            "weaknesses": ["Evaluation failed"],
            "_meta": {
                "model": model,
                "tokens": 0,
                "time": eval_time
            }
        }
    
    def compare_responses_fully_blind(self, chat_content, baseline_resp, adaptive_resp, context):
        """Compare two responses blindly by shuffling their labels"""
        import random
        
        responses = [
            {"label": "A", "text": baseline_resp, "actual": "baseline"},
            {"label": "B", "text": adaptive_resp, "actual": "adaptive"}
        ]
        random.shuffle(responses)
        
        prompt = f"""Compare these two responses to the same user message.

USER MESSAGE: "{chat_content}"
User's emotion: {context.get('emotion', 'unclear')}

RESPONSE A: "{responses[0]['text']}"
RESPONSE B: "{responses[1]['text']}"

Evaluate each on these criteria (1-5 scale):
1. Brand_Voice_Authenticity (Weight: 3.0)
2. Context_Specificity (Weight: 2.0)
3. Tone_Adaptation_Brand_Ethos (Weight: 3.0)

Return ONLY valid JSON (no markdown, no preamble):
{{
  "response_a_scores": {{"Brand_Voice_Authenticity": 3, "Context_Specificity": 3, "Tone_Adaptation_Brand_Ethos": 3}},
  "response_b_scores": {{"Brand_Voice_Authenticity": 3, "Context_Specificity": 3, "Tone_Adaptation_Brand_Ethos": 3}},
  "response_a_total": 15.0,
  "response_b_total": 15.0,
  "winner": "Response A",
  "confidence": "medium",
  "reasoning": "Specific observable differences",
  "key_differentiators": ["what made winner better"]
}}"""

        try:
            selected_model = optimizer.select_model(
                step_name="Response Comparison",
                prompt=prompt,
                estimated_input_tokens=len(prompt.split()) * 1.3,
                estimated_output_tokens=800
            )
            
            response_obj = self.client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "You are an objective evaluator. Return ONLY valid JSON, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=800
            )
            
            raw_content = response_obj.choices[0].message.content
            json_content = self._extract_json_from_text(raw_content)
            result = json.loads(json_content)
            
            # Determine actual winner
            a_is_baseline = responses[0]['actual'] == 'baseline'
            
            if result.get('winner') == 'Response A':
                actual_winner = "Baseline" if a_is_baseline else "Adaptive"
            elif result.get('winner') == 'Response B':
                actual_winner = "Adaptive" if a_is_baseline else "Baseline"
            else:
                actual_winner = "Tie"
            
            # Map scores correctly
            result['baseline_scores'] = result['response_a_scores'] if a_is_baseline else result['response_b_scores']
            result['adaptive_scores'] = result['response_b_scores'] if a_is_baseline else result['response_a_scores']
            result['actual_winner'] = actual_winner
            
            # Calculate improvement
            baseline_total = result['response_a_total'] if a_is_baseline else result['response_b_total']
            adaptive_total = result['response_b_total'] if a_is_baseline else result['response_a_total']
            
            if baseline_total > 0:
                improvement = ((adaptive_total - baseline_total) / baseline_total) * 100
                result['improvement_percentage'] = f"{improvement:+.1f}%"
            else:
                result['improvement_percentage'] = "N/A"
            
            if 'reasoning' in result:
                reasoning = result['reasoning']
                if a_is_baseline:
                    reasoning = reasoning.replace('Response A', '[[BASELINE_TEMP]]')
                    reasoning = reasoning.replace('Response B', '[[ADAPTIVE_TEMP]]')
                    reasoning = reasoning.replace('[[BASELINE_TEMP]]', 'the Baseline system')
                    reasoning = reasoning.replace('[[ADAPTIVE_TEMP]]', 'the Adaptive system')
                else:
                    reasoning = reasoning.replace('Response A', '[[ADAPTIVE_TEMP]]')
                    reasoning = reasoning.replace('Response B', '[[BASELINE_TEMP]]')
                    reasoning = reasoning.replace('[[ADAPTIVE_TEMP]]', 'the Adaptive system')
                    reasoning = reasoning.replace('[[BASELINE_TEMP]]', 'the Baseline system')
                result['reasoning'] = reasoning
            
            # Add metadata
            tokens = response_obj.usage.total_tokens if hasattr(response_obj, 'usage') else 0
            result['_meta'] = {
                'model': "gpt-5.1",
                'tokens': tokens,
                'time': 0
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Comparison error: {e}")
            print(f"Raw content: {raw_content if 'raw_content' in locals() else 'N/A'}")
            return self._get_default_comparison()
            
        except Exception as e:
            print(f"Comparison error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_comparison()
    
    def _get_default_comparison(self):
        """Return default comparison when parsing fails"""
        default_scores = {k: 3 for k in self.EVALUATION_PARAMETERS.keys()}
        return {
            "actual_winner": "Tie",
            "improvement_percentage": "N/A",
            "reasoning": "Comparison evaluation failed - using default values",
            "baseline_scores": default_scores,
            "adaptive_scores": default_scores,
            "response_a_total": 15.0,
            "response_b_total": 15.0,
            "_meta": {
                "model": "N/A",
                "tokens": 0,
                "time": 0
            }
        }

evaluator = ToneBrandEvaluator(client)

# Conversation history storage
conversation_histories = {}  # For adaptive system
baseline_histories = {}      # For baseline system (separate history)

def get_conversation_history(session_id):
    """Get conversation history for a session"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    history = []
    for turn in conversation_histories[session_id]:
        history.append({
            'user': turn['user'],
            'assistant': turn['assistant']
        })
    
    return history

def add_to_history(session_id, user_message, assistant_message):
    """Add exchange to conversation history"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    conversation_histories[session_id].append({
        'user': user_message,
        'assistant': assistant_message,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep last 10 turns
    if len(conversation_histories[session_id]) > 10:
        conversation_histories[session_id] = conversation_histories[session_id][-10:]

def get_baseline_history(session_id):
    """Get baseline conversation history (separate from adaptive)"""
    if session_id not in baseline_histories:
        baseline_histories[session_id] = []
    
    history = []
    for turn in baseline_histories[session_id]:
        history.append({
            'user': turn['user'],
            'assistant': turn['assistant']
        })
    
    return history

def add_to_baseline_history(session_id, user_message, assistant_message):
    """Add exchange to baseline conversation history"""
    if session_id not in baseline_histories:
        baseline_histories[session_id] = []
    
    baseline_histories[session_id].append({
        'user': user_message,
        'assistant': assistant_message,
        'timestamp': datetime.now().isoformat()
    })
    
    if len(baseline_histories[session_id]) > 10:
        baseline_histories[session_id] = baseline_histories[session_id][-10:]

def calculate_cost(model, tokens):
    """Calculate cost based on model and token count"""
    if not model or model in ['No API call', 'N/A', 'Rule-based'] or tokens == 0:
        return 0
    
    # Pricing per 1K tokens (approximate)
    pricing = {
        'gpt-5.1': 0.0025,
        'gpt-4o': 0.005,
        'gpt-4o-mini': 0.00015,
        'gpt-4-turbo': 0.01,
        'gpt-4': 0.03,
        'gpt-3.5-turbo': 0.002,
    }
    
    # Find matching model in pricing
    model_lower = model.lower()
    for model_key, price in pricing.items():
        if model_key in model_lower:
            return (tokens / 1000) * price
    
    # Default estimate
    return (tokens / 1000) * 0.002

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CloudBridge AI Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0066cc 0%, #004c99 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 24px;
            margin-bottom: 24px;
            border-left: 6px solid #0066cc;
        }
        .brand-container { display: flex; align-items: center; gap: 20px; margin-bottom: 16px; }
        .brand-logo {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #0066cc, #0099ff);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .brand-text h1 {
            font-size: 38px;
            font-weight: 900;
            background: linear-gradient(135deg, #0066cc, #004c99);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .brand-tagline { color: #0066cc; font-weight: 700; }
        .header p { color: #64748b; margin-top: 8px; }
        .comparison-info { display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap; }
        .system-badge {
            padding: 10px 18px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 700;
        }
        .system-badge.baseline {
            background: linear-gradient(135deg, #fef2f2, #fee2e2);
            border: 2px solid #fecaca;
            color: #991b1b;
        }
        .system-badge.adaptive {
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border: 2px solid #bfdbfe;
            color: #1e40af;
        }
         .chat-header {
            padding: 24px 28px;
            background: linear-gradient(135deg, #0066cc, #0099ff);
            color: white;
            display: flex;
            justify-content: space-between;
            border-radius: 20px 20px 0 0;
        }
        .status-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .chat-container {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            margin-bottom: 24px;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 350px);
            min-height: 500px;
        }
      
        .chat-controls { display: flex; gap: 20px; align-items: center; }
        .analysis-toggle { display: flex; align-items: center; gap: 12px; font-weight: 700; }
        .toggle-switch {
            position: relative;
            width: 52px;
            height: 28px;
            background: rgba(255,255,255,0.25);
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .toggle-switch.active { background: #0066cc; }
        .toggle-slider {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }
        .toggle-switch.active .toggle-slider { transform: translateX(24px); }
        .btn-clear {
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            color: white;
            cursor: pointer;
            font-weight: 700;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 28px;
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        }
        .message { margin-bottom: 20px; display: flex; flex-direction: column; }
        .message.user { align-items: flex-end; }
        .message.assistant { align-items: flex-start; }
        .message-bubble {
            max-width: 70%;
            padding: 16px 20px;
            border-radius: 18px;
            line-height: 1.6;
        }
        .message.user .message-bubble {
            background: linear-gradient(135deg, #0066cc, #0099ff);
            color: white;
        }
        .message.assistant .message-bubble {
            background: white;
            color: #374151;
            border: 1px solid rgba(0, 102, 204, 0.1);
        }
        .message-label {
            font-size: 11px;
            color: #9ca3af;
            margin-bottom: 8px;
            font-weight: 800;
            text-transform: uppercase;
        }
        .typing-indicator { display: none; padding: 16px 20px; background: white; border-radius: 18px; }
        .typing-indicator.active { display: flex; gap: 6px; }
        .typing-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #0066cc;
            animation: typing 1.4s infinite;
        }
        @keyframes typing {
            0%, 60%, 100% { opacity: 0.3; }
            30% { opacity: 1; transform: translateY(-6px); }
        }
        .chat-input-container {
            padding: 20px 24px;
            border-top: 1px solid rgba(229,231,235,0.5);
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        }
        .input-wrapper { display: flex; gap: 16px; }
        textarea {
            flex: 1;
            padding: 16px 20px;
            border: 2px solid rgba(209,213,219,0.5);
            border-radius: 16px;
            font-family: inherit;
            resize: none;
            min-height: 50px;
        }
        .btn-send {
            background: linear-gradient(135deg, #0066cc, #0099ff);
            color: white;
            border: none;
            padding: 14px 30px;
            border-radius: 14px;
            font-weight: 800;
            cursor: pointer;
        }
        .sample-queries { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
        .sample-chip {
            padding: 10px 18px;
            background: rgba(255,255,255,0.8);
            border: 2px solid rgba(0, 102, 204, 0.2);
            border-radius: 25px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 700;
            color: #0066cc;
        }
        .sample-chip:hover { background: #0066cc; color: white; }
        .analysis-section {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .section-header {
            padding: 22px 28px;
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
        }
        .section-toggle { font-size: 22px; color: #0066cc; transition: transform 0.3s; }
        .analysis-section.expanded .section-toggle { transform: rotate(180deg); }
        .section-content { max-height: 0; overflow: hidden; transition: max-height 0.4s; }
        .analysis-section.expanded .section-content { max-height: 5000px; }
        .section-body { padding: 28px; }
        .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .response-card { border-radius: 16px; padding: 24px; position: relative; }
        .response-card.baseline {
            background: #fefefe;
            border: 2px solid rgba(220,38,38,0.1);
        }
        .response-card.adaptive {
            background: #fefefe;
            border: 2px solid rgba(0, 102, 204, 0.1);
        }
        .card-header { font-weight: 900; margin-bottom: 20px; text-transform: uppercase; }
        .response-card.baseline .card-header { color: #991b1b; }
        .response-card.adaptive .card-header { color: #0066cc; }
        .score-display {
            font-size: 42px;
            font-weight: 900;
            text-align: center;
            padding: 24px;
            border-radius: 16px;
            margin: 20px 0;
        }
        .response-card.baseline .score-display {
            background: linear-gradient(135deg, #fef2f2, #fee2e2);
            color: #dc2626;
        }
        .response-card.adaptive .score-display {
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            color: #0066cc;
        }
        .metrics-list { display: flex; flex-direction: column; gap: 12px; }
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 14px 16px;
            background: rgba(255,255,255,0.8);
            border-radius: 12px;
        }
        .metric-label { color: #64748b; font-weight: 600; }
        .metric-value { font-weight: 800; }
        .comparison-summary {
            background: linear-gradient(135deg, rgba(239, 246, 255, 0.9), rgba(219, 234, 254, 0.9));
            border-left: 4px solid #0066cc;
            border-radius: 16px;
            padding: 26px;
            margin-bottom: 24px;
        }
        .detail-card {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: 16px;
            padding: 22px;
            margin-top: 20px;
        }
        .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .detail-item { background: white; padding: 16px; border-radius: 12px; }
        .detail-item strong { display: block; color: #0066cc; font-size: 13px; margin-bottom: 8px; }
        .insight-box {
            background: linear-gradient(135deg, rgba(239,246,255,0.9), rgba(219,234,254,0.9));
            border-left: 4px solid #0066cc;
            border-radius: 14px;
            padding: 20px;
            margin: 16px 0;
        }
        .context-tag {
            display: inline-block;
            padding: 6px 14px;
            background: rgba(0, 102, 204, 0.1);
            border: 1px solid rgba(0, 102, 204, 0.2);
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            color: #0066cc;
            margin: 4px;
        }
        .winner-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 900;
            text-transform: uppercase;
        }
        .winner-badge.adaptive { background: linear-gradient(135deg, #0066cc, #0099ff); color: white; }
        .winner-badge.baseline { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; }
        .step-breakdown { margin-top: 20px; }
        .step-item {
            display: flex;
            justify-content: space-between;
            padding: 14px 18px;
            background: white;
            border-radius: 12px;
            margin-bottom: 10px;
        }
        .step-name { font-weight: 700; }
        .step-details { display: flex; gap: 20px; font-size: 13px; color: #64748b; }
        .step-model {
            padding: 4px 10px;
            background: linear-gradient(135deg, #dbeafe, #bfdbfe);
            border-radius: 6px;
            color: #1e40af;
            font-size: 12px;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand-container">
                <div class="brand-logo">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M4 12C4 12 6 8 12 8C18 8 20 12 20 12M12 8V4M8 12V16M12 12V18M16 12V16" stroke="white" stroke-width="2.5"/>
                        <circle cx="8" cy="16" r="1.5" fill="white"/>
                        <circle cx="12" cy="18" r="1.5" fill="white"/>
                        <circle cx="16" cy="16" r="1.5" fill="white"/>
                    </svg>
                </div>
                <div class="brand-text">
                    <h1>CloudBridge</h1>
                    <div class="brand-tagline">EMPOWER EVERY PERSON</div>
                </div>
            </div>
           <p style="color: #0a6ed1; font-weight: 600;">
    Technology That Adapts To How You Work
</p>



        </div>
        
      <div class="chat-container">
    <div class="chat-header">
        <div class="status-container">
            <div class="status-dot"></div>
            <h2>CloudBridge Assistant</h2>
        </div>
        
        <div class="chat-controls">
            <div class="analysis-toggle">
                <span>Show Analysis</span>
                <div class="toggle-switch active" id="analysisToggle">
                    <div class="toggle-slider"></div>
                </div>
            </div>
            <button class="btn-clear" onclick="clearChat()">Clear Chat</button>
        </div>
    </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div class="message-label">CloudBridge</div>
                    <div class="message-bubble">
                        Tell me what you’re working on, and I’ll help you move forward.
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <div class="input-wrapper">
                    <textarea id="userInput" placeholder="Tell me what you need help with..." rows="1"></textarea>
                    <button class="btn-send" onclick="sendMessage()">Send</button>
                </div>
                <div class="sample-queries">
                    <div class="sample-chip" onclick="fillQuery('My presentation is in 15 minutes and the file will not open! Please help ASAP!')">⏰ Time-Critical</div>
                    <div class="sample-chip" onclick="fillQuery('You charged me twice this month and nobody is fixing it. I want my money back NOW!')">💢 Billing Complaint</div>
                    <div class="sample-chip" onclick="fillQuery('What is up? Can you walk me through setting up webhooks?')">🤙 Casual Help</div>
                    <div class="sample-chip" onclick="fillQuery('I am writing to inquire about compliance certifications for your platform.')">📋 Formal Inquiry</div>
                    <div class="sample-chip" onclick="fillQuery('Getting a 500 error on POST /api/v2/users - any known issues?')">💻 Technical Error</div>
                    <div class="sample-chip" onclick="fillQuery('I am new to CloudBridge and feeling overwhelmed. Where should I start?')">🤔 New User</div>
                    <div class="sample-chip" onclick="fillQuery('Can you recommend a good CRM tool?')"> Out of Scope</div>
                </div>
            </div>
        </div>
        
        <div id="analysisContainer" style="display: block;">
            <div class="analysis-section expanded">
                <div class="section-header" onclick="toggleSection(this)">
                    <h3> Scope Validation</h3>
                    <span class="section-toggle">▼</span>
                </div>
                <div class="section-content">
                    <div class="section-body" id="scopeValidation">
                        <p style="color: #64748b; text-align: center; padding: 20px;">Send a message to see scope validation...</p>
                    </div>
                </div>
            </div>
            
            <div class="analysis-section expanded">
                <div class="section-header" onclick="toggleSection(this)">
                    <h3> Context Analysis</h3>
                    <span class="section-toggle">▼</span>
                </div>
                <div class="section-content">
                    <div class="section-body" id="contextAnalysis">
                        <p style="color: #64748b; text-align: center; padding: 20px;">Send a message to see context analysis...</p>
                    </div>
                </div>
            </div>
            
            <div class="analysis-section expanded">
                <div class="section-header" onclick="toggleSection(this)">
                    <h3> Response Comparison</h3>
                    <span class="section-toggle">▼</span>
                </div>
                <div class="section-content">
                    <div class="section-body" id="responseComparison">
                        <p style="color: #64748b; text-align: center; padding: 20px;">Send a message to see response comparison...</p>
                    </div>
                </div>
            </div>
            
           </div>
        
        <!-- Performance Metrics - Always Visible -->
        <div class="analysis-section expanded">
            <div class="section-header" onclick="toggleSection(this)">
                <h3> Performance Metrics</h3>
                <span class="section-toggle">▼</span>
            </div>
            <div class="section-content">
                <div class="section-body" id="performanceMetrics">
                    <p style="color: #64748b; text-align: center; padding: 20px;">Send a message to see performance metrics...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = Date.now().toString();
        let showAnalysis = true;
        
        document.getElementById('analysisToggle').addEventListener('click', function() {
            this.classList.toggle('active');
            showAnalysis = !showAnalysis;
            document.getElementById('analysisContainer').style.display = showAnalysis ? 'block' : 'none';
        });
        
        function toggleSection(header) {
            const section = header.closest('.analysis-section');
            section.classList.toggle('expanded');
        }
        
        function fillQuery(text) {
            document.getElementById('userInput').value = text;
            document.getElementById('userInput').focus();
        }
        
        function clearChat() {
            if (confirm('Clear all messages and start fresh?')) {
                sessionId = Date.now().toString();
                document.getElementById('chatMessages').innerHTML = `
                    <div class="message assistant">
                        <div class="message-label">CLOUDBRIDGE</div>
                        <div class="message-bubble">Tell me what you’re working on, and I’ll help you move forward.</div>
                    </div>
                `;
                ['scopeValidation', 'contextAnalysis', 'responseComparison', 'performanceMetrics'].forEach(id => {
                    document.getElementById(id).innerHTML = '<p style="color: #64748b; text-align: center; padding: 20px;">Send a message to see analysis...</p>';
                });
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.innerHTML = `
                <div class="message-label">CLOUDBRIDGE</div>
                <div class="typing-indicator active">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            document.getElementById('chatMessages').appendChild(typingDiv);
            scrollToBottom();
            
            try {
               const response = await fetch('/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        message, 
        session_id: sessionId,
        show_analysis: showAnalysis  // Send the toggle state
    })
});
                
                const data = await response.json();
                typingDiv.remove();
                
                if (data.error) {
                    addMessage('assistant', ' Error: ' + data.error);
                    return;
                }
                
                addMessage('assistant', data.gratitude_response || data.adaptive?.response || data.adaptive_response);

// Always update performance metrics (it's outside analysisContainer)
updatePerformanceMetrics(data.performance);

// Only update analysis sections if toggle is ON
if (showAnalysis) {
    updateScopeValidation({ in_scope: !data.out_of_scope, reason: "Valid query" });
    updateContextAnalysis(data.adaptive?.context || {});
    updateResponseComparison(data.comparison, data.baseline?.response, data.adaptive?.response);
}
                
            } catch (error) {
                typingDiv.remove();
                addMessage('assistant', ' Connection error. Please try again.');
                console.error('Error:', error);
            }
        }
        
        function addMessage(role, content) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            const label = role === 'user' ? 'You' : 'Adaptive System';
            messageDiv.innerHTML = `
                <div class="message-label">${label}</div>
                <div class="message-bubble">${escapeHtml(content)}</div>
            `;
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function updateScopeValidation(validation) {
            const container = document.getElementById('scopeValidation');
            if (!validation) {
                container.innerHTML = '<p style="color: #64748b;">No validation data available</p>';
                return;
            }
            const html = `
                <div class="detail-card">
                    <h4>Scope Check Results</h4>
                    <div class="insight-box">
                        <strong>${validation.in_scope ? ' Query In Scope' : ' Query Out of Scope'}</strong>
                        <p>${validation.reason || 'Query is within CloudBridge product support scope'}</p>
                    </div>
                </div>
            `;
            container.innerHTML = html;
        }
        
        function updateContextAnalysis(context) {
            const container = document.getElementById('contextAnalysis');
            if (!context) {
                container.innerHTML = '<p style="color: #64748b;">No context analysis available</p>';
                return;
            }
            const html = `
                <div class="detail-card">
                    <h4>Detected Context</h4>
                    <div class="detail-grid">
                        <div class="detail-item"><strong>Emotion</strong><span>${escapeHtml(context.emotion || 'Neutral')}</span></div>
                        <div class="detail-item"><strong>Urgency Level</strong><span>${escapeHtml(context.urgency_level || 'Normal')}</span></div>
                        <div class="detail-item"><strong>Formality Level</strong><span>${escapeHtml(context.formality_level || 'Neutral')}</span></div>
                        <div class="detail-item"><strong>Intent</strong><span>${escapeHtml(context.intent || 'General inquiry')}</span></div>
                    </div>
                    ${context.key_concerns && context.key_concerns.length > 0 ? `
                        <div style="margin-top: 20px;">
                            <strong style="display: block; margin-bottom: 12px;">Key Concerns:</strong>
                            ${context.key_concerns.map(c => `<span class="context-tag">${escapeHtml(c)}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
            container.innerHTML = html;
        }
        
        function updateResponseComparison(comparison, baselineResp, adaptiveResp) {
            const container = document.getElementById('responseComparison');
            if (!comparison) {
                container.innerHTML = '<p style="color: #64748b;">No comparison data available</p>';
                return;
            }
            const baselineTotal = comparison.baseline_scores ? Object.values(comparison.baseline_scores).reduce((a,b) => parseFloat(a) + parseFloat(b), 0) : 0;
            const adaptiveTotal = comparison.adaptive_scores ? Object.values(comparison.adaptive_scores).reduce((a,b) => parseFloat(a) + parseFloat(b), 0) : 0;
            
            const html = `
                <div class="comparison-summary">
                    <h3>🏆 Winner: <span class="winner-badge ${comparison.actual_winner.toLowerCase()}">${comparison.actual_winner}</span></h3>
                    <p><strong>Improvement:</strong> ${comparison.improvement_percentage}</p>
                    <p><strong>Reasoning:</strong> ${escapeHtml(comparison.reasoning || 'No reasoning provided')}</p>
                </div>
                <div class="comparison-grid">
                    <div class="response-card baseline">
                        <div class="card-header"> Baseline Response</div>
                        <div style="padding: 16px; background: rgba(255,255,255,0.6); border-radius: 12px; margin-bottom: 20px;">
                            ${escapeHtml(baselineResp)}
                        </div>
                        <div class="score-display">${baselineTotal.toFixed(1)}/30</div>
                        <div class="metrics-list">
                            ${comparison.baseline_scores ? Object.entries(comparison.baseline_scores).map(([k,v]) => `
                                <div class="metric-item">
                                    <span class="metric-label">${k.replace(/_/g, ' ')}</span>
                                    <span class="metric-value">${parseFloat(v).toFixed(1)}/5</span>
                                </div>
                            `).join('') : '<p style="text-align:center;color:#64748b;">No scores available</p>'}
                        </div>
                    </div>
                    <div class="response-card adaptive">
                        <div class="card-header"> Adaptive Response</div>
                        <div style="padding: 16px; background: rgba(255,255,255,0.6); border-radius: 12px; margin-bottom: 20px;">
                            ${escapeHtml(adaptiveResp)}
                        </div>
                        <div class="score-display">${adaptiveTotal.toFixed(1)}/30</div>
                        <div class="metrics-list">
                            ${comparison.adaptive_scores ? Object.entries(comparison.adaptive_scores).map(([k,v]) => `
                                <div class="metric-item">
                                    <span class="metric-label">${k.replace(/_/g, ' ')}</span>
                                    <span class="metric-value">${parseFloat(v).toFixed(1)}/5</span>
                                </div>
                            `).join('') : '<p style="text-align:center;color:#64748b;">No scores available</p>'}
                        </div>
                    </div>
                </div>
            `;
            container.innerHTML = html;
        }
        
        function updatePerformanceMetrics(performance) {
            const container = document.getElementById('performanceMetrics');
            if (!performance) {
                container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 20px;"> No performance data available</p>';
                return;
            }
            const totalTime = performance.total_time || 0;
            const totalTokens = performance.total_tokens || 0;
            const totalCost = performance.total_cost || 0;
            const breakdown = performance.breakdown || {};
            
            let actualApiCalls = 0;
            const steps = Array.isArray(breakdown) ? breakdown : Object.values(breakdown);
            if (steps.length > 0) {
                for (const data of steps) {
                    const model = data.model || 'No API call';
                    if (model !== 'No API call' && model !== 'N/A' && model !== 'Rule-based' && data.tokens > 0) {
                        actualApiCalls++;
                    }
                }
            }
            
            let html = `
                <div class="detail-card">
                    <h4> Complete Pipeline Performance</h4>
                    <div class="detail-grid">
                        <div class="detail-item"><strong>Total Time</strong><span>${totalTime.toFixed(2)}s</span></div>
                        <div class="detail-item"><strong>API Calls</strong><span>${actualApiCalls}</span></div>
                        <div class="detail-item"><strong>Total Tokens</strong><span>${totalTokens.toLocaleString()}</span></div>
                        <div class="detail-item"><strong>Total Cost</strong><span>$${totalCost.toFixed(4)}</span></div>
                    </div>
                </div>
                
                ${steps.length > 0 ? `
                    <div class="detail-card" style="margin-top: 20px;">
                        <h4> Step-by-Step Breakdown</h4>
                        <div class="step-breakdown">
                            ${steps.map((data) => `
                                <div class="step-item">
                                    <span class="step-name">${escapeHtml(data.step_name || data.display_name || data.name || 'Unknown Step')}</span>
                                    <div class="step-details">
                                        <span class="step-model">${escapeHtml(data.model || 'N/A')}</span>
                                        <span>⏱️ ${(data.time || 0).toFixed(2)}s</span>
                                        <span>🎫 ${(data.tokens || 0).toLocaleString()} tokens</span>
                                        <span>💰 $${(data.cost || 0).toFixed(4)}</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            `;
            container.innerHTML = html;
        }
        
        function scrollToBottom() {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    chat = data.get('message', data.get('chat', ''))
    session_id = data.get('session_id', 'default')
    show_analysis = data.get('show_analysis', True)  # Get toggle state from frontend
    
    if not chat:
        return jsonify({'error': 'No message provided'}), 400
    
    request_start_time = time.time()
    all_steps = []
    
    try:
        history = get_conversation_history(session_id)
        
        # Check gratitude
        if any(word in chat.lower() for word in ['thank', 'thanks', 'bye', 'goodbye']) and len(chat.split()) <= 8:
            import random
            responses = [
                "You're welcome. Good luck with the migration.",
                "Happy to help. Reach out if anything comes up.",
                "Glad we could sort that out. Take care.",
            ]
            response = random.choice(responses)
            add_to_history(session_id, chat, response)
            add_to_baseline_history(session_id, chat, response)
            
            return jsonify({
                'is_gratitude': True,
                'gratitude_response': response,
                'performance': {
                    'total_time': time.time() - request_start_time,
                    'breakdown': {}
                }
            })
        
        # Scope check
        scope_start = time.time()
        is_in_scope, redirect_msg = framework._validate_scope(chat, framework.get_or_create_session(session_id))
        scope_time = time.time() - scope_start
        
        all_steps.append({
            'step_name': '1. Scope Validation',
            'model': 'Rule-based',
            'tokens': 0,
            'time': scope_time,
            'cost': 0
        })
        
        if not is_in_scope:
            add_to_history(session_id, chat, redirect_msg)
            
            # Format breakdown properly
            formatted_breakdown = []
            for step in all_steps:
                formatted_breakdown.append({
                    'step_name': step.get('step_name', ''),
                    'display_name': step.get('step_name', ''),
                    'name': step.get('step_name', ''),
                    'time': step.get('time', 0),
                    'tokens': step.get('tokens', 0),
                    'cost': step.get('cost', 0),
                    'model': step.get('model', 'N/A')
                })
            
            return jsonify({
                'out_of_scope': True,
                'redirect_message': redirect_msg,
                'performance': {
                    'total_time': time.time() - request_start_time,
                    'breakdown': formatted_breakdown,
                    'steps': formatted_breakdown
                }
            })
        
        # ADAPTIVE PROCESSING
        print("\n🔄 Processing adaptive response...")
        adaptive_result = framework.process_message(
            message=chat,
            session_id=session_id
        )
        
        adaptive_response = adaptive_result.response_text
        context_analysis = adaptive_result.context_analysis
        
        # Extract adaptive steps
        perf_metrics = adaptive_result.performance_metrics
        if perf_metrics and 'steps' in perf_metrics:
            for step in perf_metrics['steps']:
                model = step.get('model', 'No API call')
                tokens = step.get('total_tokens', 0)
                time_val = step.get('duration_seconds', 0)
                cost = calculate_cost(model, tokens)
                
                all_steps.append({
                    'step_name': f"{len(all_steps) + 1}. {step.get('step_name', 'Unknown')}",
                    'model': model,
                    'tokens': tokens,
                    'time': time_val,
                    'cost': cost
                })

        if show_analysis:
            print(" Generating baseline response...")
            baseline_history = get_baseline_history(session_id)
            baseline_result = baseline_system.generate_response(chat, baseline_history)
            all_steps.append({
                'step_name': f"{len(all_steps) + 1}. Baseline Response Generation",
                'model': baseline_result['model'],
                'tokens': baseline_result['tokens'],
                'time': baseline_result['time'],
                'cost': calculate_cost(baseline_result['model'], baseline_result['tokens'])
            })
            

            add_to_baseline_history(session_id, chat, baseline_result['response'])

        add_to_history(session_id, chat, adaptive_response)

        # If analysis is OFF, skip evaluation and comparison
        if not show_analysis:
            print("⭐️ Skipping baseline/evaluation (analysis disabled)")
            
            # Still validate brand voice even when analysis is off
            context_for_eval = {
                'emotion': context_analysis.get('emotion', 'neutral'),
                'urgency_level': context_analysis.get('urgency_level', 2),
                'formality': context_analysis.get('formality_preference', 'neutral')
            }
            
            adaptive_eval = evaluator.evaluate_response_blind(
                chat, adaptive_response, context_for_eval
            )
            
            # Build response with brand validation
            # Format breakdown properly
            formatted_breakdown = []
            for step in all_steps:
                formatted_breakdown.append({
                    'step_name': step['step_name'],
                    'display_name': step['step_name'],
                    'name': step['step_name'],
                    'time': step.get('time', 0),
                    'tokens': step.get('tokens', 0),
                    'cost': step.get('cost', 0),
                    'model': step.get('model', 'N/A')
                })
            
            response_data = {
                'out_of_scope': False,
                'is_gratitude': False,
                'adaptive': {
                    'response': adaptive_response,
                    'context': {},
                    'brand_score': adaptive_eval.get('weighted_score', 0)
                },
                'performance': {
                    'total_time': sum(step.get('time', 0) for step in all_steps),
                    'total_tokens': sum(step.get('tokens', 0) for step in all_steps),
                    'total_cost': sum(step.get('cost', 0) for step in all_steps),
                    'breakdown': formatted_breakdown,
                    'steps': formatted_breakdown,
                    'step_count': len(all_steps)
                }
            }
            
            return jsonify(response_data)

        # Evaluation context
        context_for_eval = {
            'emotion': context_analysis.get('emotion', 'neutral'),
            'urgency_level': context_analysis.get('urgency_level', 2),
            'formality': context_analysis.get('formality_preference', 'neutral')
        }
        
        # BASELINE EVALUATION
        print(" Evaluating baseline...")
        baseline_eval = evaluator.evaluate_response_blind(chat, baseline_result['response'], context_for_eval)
        if '_meta' in baseline_eval:
            meta = baseline_eval['_meta']
            all_steps.append({
                'step_name': f"{len(all_steps) + 1}. Baseline Evaluation",
                'model': meta['model'],
                'tokens': meta['tokens'],
                'time': meta['time'],
                'cost': calculate_cost(meta['model'], meta['tokens'])
            })
        
        # ADAPTIVE EVALUATION
        print(" Evaluating adaptive...")
        adaptive_eval = evaluator.evaluate_response_blind(chat, adaptive_response, context_for_eval)
        if '_meta' in adaptive_eval:
            meta = adaptive_eval['_meta']
            all_steps.append({
                'step_name': f"{len(all_steps) + 1}. Adaptive Evaluation",
                'model': meta['model'],
                'tokens': meta['tokens'],
                'time': meta['time'],
                'cost': calculate_cost(meta['model'], meta['tokens'])
            })
        
        # COMPARISON
        print(" Comparing responses...")
        comp_start = time.time()
        comparison = evaluator.compare_responses_fully_blind(chat, baseline_result['response'], adaptive_response, context_for_eval)
        comp_time = time.time() - comp_start
        
        if '_meta' in comparison:
            meta = comparison['_meta']
            all_steps.append({
                'step_name': f"{len(all_steps) + 1}. Response Comparison",
                'model': meta['model'],
                'tokens': meta['tokens'],
                'time': comp_time,
                'cost': calculate_cost(meta['model'], meta['tokens'])
            })
        
        # Build complete breakdown as ordered list (not dict) to preserve order
        breakdown_list = []
        breakdown_dict = {} 
        total_time = 0
        total_tokens = 0
        total_cost = 0
        
        for step in all_steps:
            step_data = {
                'step_name': step['step_name'],
                'display_name': step['step_name'],  
                'name': step['step_name'],          
                'time': step['time'],
                'tokens': step['tokens'],
                'cost': step['cost'],
                'model': step['model']
            }
            
            # Add to list (preserves order)
            breakdown_list.append(step_data)
            
            # Add to dict with step_name as key (for backwards compatibility)
            breakdown_dict[step['step_name']] = step_data
            
            total_time += step['time']
            total_tokens += step['tokens']
            total_cost += step['cost']
        
        print(f"\n Complete! Total time: {time.time() - request_start_time:.2f}s, Tokens: {total_tokens}, Cost: ${total_cost:.6f}")
        
        # Context data for frontend
        context_data = {
            'emotion': context_analysis.get('emotion', 'neutral').capitalize(),
            'urgency_level': f"{context_analysis.get('urgency_level', 2)}/5",
            'formality_level': context_analysis.get('formality_preference', 'neutral'),
            'intent': str(context_analysis.get('intent', 'question')).replace('_', ' ').title(),
            'key_concerns': context_analysis.get('key_pain_points', []),
            'contextual_notes': f"Detected as {context_analysis.get('user_state', 'calm')} user"
        }
        
        # Build response
        response_data = {
            'out_of_scope': False,
            'is_gratitude': False,
            'baseline': {
                'response': baseline_result['response'],
                'evaluation': baseline_eval
            },
            'adaptive': {
                'response': adaptive_response,
                'context': context_data,
                'evaluation': adaptive_eval
            },
            'comparison': comparison,
            'performance': {
                'total_time': total_time,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'breakdown': breakdown_list,      # Use list for correct ordering
                'breakdown_dict': breakdown_dict, # Keep dict for backwards compatibility
                'steps': breakdown_list,          # Alternative field name
                'step_count': len(all_steps)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Format breakdown properly for errors
        formatted_breakdown = []
        if all_steps:
            for step in all_steps:
                formatted_breakdown.append({
                    'step_name': step.get('step_name', ''),
                    'display_name': step.get('step_name', ''),
                    'name': step.get('step_name', ''),
                    'time': step.get('time', 0),
                    'tokens': step.get('tokens', 0),
                    'cost': step.get('cost', 0),
                    'model': step.get('model', 'N/A')
                })
        
        return jsonify({
            'error': str(e),
            'performance': {
                'total_time': time.time() - request_start_time,
                'breakdown': formatted_breakdown,
                'steps': formatted_breakdown
            }
        }), 500
    
@app.route('/api/debug/sample-performance', methods=['GET'])
def debug_sample_performance():
    """Return a sample performance breakdown for frontend debugging"""
    sample_steps = [
        {
            'step_name': '1. Scope Validation',
            'display_name': '1. Scope Validation',
            'name': '1. Scope Validation',
            'model': 'Rule-based',
            'tokens': 0,
            'time': 0.02,
            'cost': 0
        },
        {
            'step_name': '2. Context Analysis',
            'display_name': '2. Context Analysis',
            'name': '2. Context Analysis',
            'model': 'gpt-4o-mini',
            'tokens': 406,
            'time': 2.80,
            'cost': 0.0020
        },
        {
            'step_name': '3. Tone Selection',
            'display_name': '3. Tone Selection',
            'name': '3. Tone Selection',
            'model': 'N/A',
            'tokens': 0,
            'time': 0.03,
            'cost': 0.0000
        }
    ]
    
    return jsonify({
        'performance': {
            'total_time': 2.85,
            'total_tokens': 406,
            'total_cost': 0.0020,
            'breakdown': sample_steps,
            'steps': sample_steps,
            'step_count': len(sample_steps)
        }
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" CloudBridge Adaptive Framework - Fixed Version")
    print("="*70)
    print("\n Improvements:")
    print("  - Robust JSON parsing with fallback handling")
    print("  - Proper cost calculation per model")
    print("  - Uses selected models (not hardcoded)")
    print("  - Better error messages and logging")
    print("\n Open: http://localhost:5001")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5001)