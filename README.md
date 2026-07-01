# Brand-Consistent Adaptive AI Chatbot Framework

A brand-agnostic AI chatbot framework that adapts tone, context, and voice in real time. Every response is shaped by the brand's personality and validated against its rules — all driven by a single YAML config file. **No Python changes needed to switch brands.**

This repo contains three brand implementations built on the same shared framework:

| Brand | Industry | Personality |
|---|---|---|
| **Apex Stride** | Sports footwear | Tough coach — direct, punchy, challenge-focused |
| **Ember & Edge** | Premium kitchen knives | Patient chef-instructor — calm, sensory, craft-focused |
| **CloudBridge** | Technology / SaaS | Reliable partner — clear, empowering, jargon-free |

---

## How It Works

Every message goes through a 6-step pipeline:

```
User Message
     │
     ▼
1. Context Analysis    — detects emotion, intent, urgency
2. Special Rules Check — handles service flows (refund/replacement clarification)
3. Tone Selection      — picks the right tone profile for the situation
4. LLM Generation      — generates a brand-voice response
5. Brand Validation    — checks against forbidden phrases and linguistic rules
6. State Update        — saves session, user profile, risk scores
     │
     ▼
Validated Response  +  Callback (if service issue)
```

All rules — tone profiles, forbidden phrases, product catalog, voice guidelines, LLM models — live in `brand_config.yaml`. Swap brands by swapping the config file.

---

## Requirements

- Python 3.9+
- An OpenAI API key

```bash
pip install openai flask pyyaml python-dotenv
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Abdulmalik740/Brand_consistent_chatbot.git
cd Brand_consistent_chatbot
```

**2. Add your OpenAI key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

---

## Running Each Brand

### Apex Stride — Sports Footwear
> *"Push Beyond"* — Direct, punchy, challenge-focused. Your toughest coach.

```bash
python web_interface_universal.py    # Web UI → http://localhost:5000
python universal_adaptive_framework.py  # CLI demo
```

### Ember & Edge — Premium Kitchen Knives
> *"Where Craft Meets Kitchen"* — Calm, sensory, craft-focused. A patient chef-instructor.

```bash
python ember_edge_web.py        # Web UI → http://localhost:5002
python ember_edge_framework.py  # CLI demo
```

### CloudBridge — Technology / SaaS
> *"Empower Every Person"* — Clear, empowering, jargon-free. The reliable partner.

```bash
python web_interface_universal_cloudbridge.py        # Web UI → http://localhost:5001
python universal_adaptive_framework.py  # CLI demo
```

---

## Configuring a Brand

Everything lives in `brand_config.yaml`. Key sections:

- `brand` — name, tagline, personality description
- `voice_guidelines` — do/don't rules injected into every prompt
- `core_values` — values the LLM must apply on every response
- `products` — the only products the bot is allowed to recommend
- `tone_profiles` — tone variants (e.g. challenge, support, coach, warm)
- `tone_selection_rules` — maps situations to tone profiles
- `brand_guard` — forbidden phrases, linguistic DNA checks, context rules
- `special_rules` — service flows (refund, replacement) with callback hooks
- `llm` — models, temperature, max tokens, pricing per model


No Python changes needed — the config path is the only thing that changes.

---

## Web UI Features

Each brand's web interface includes:

- **Chat** — adaptive bot responses in real time
- **Context panel** — detected emotion, intent, motivation, urgency per message
- **Brand evaluation** — LLM-scored brand voice consistency (1–5 scale)
- **Baseline comparison** — adaptive bot vs a generic corporate bot, scored side by side
- **Performance metrics** — latency, token count, and cost broken down per pipeline step

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `BRAND_CONFIG_PATH` | No | `./brand_config.yaml` | Path to config file — override to switch brands |

## Multi-Brand Launcher
 
To run all three brands at once and switch between them with a dropdown, start each on its own port then open `multi_brand_dropdown.html` in your browser:
 
```bash
# Terminal 1
python web_interface_universal.py   # Apex Stride  → port 5000
 
# Terminal 2
python web_interface_universal_cloudbridge.py           # CloudBridge  → port 5001
 
# Terminal 3
python ember_edge_web.py            # Ember & Edge → port 5002
```
 
Then open `multi_brand_dropdown.html` in your browser. Use the dropdown in the top-right corner to switch between brands instantly.

