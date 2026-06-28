# Claude Code Guidelines (CLAUDE.md)

## Communication Style (Caveman Mode)
To reduce token consumption by ~75%, always communicate in an ultra-compressed "caveman" style:
- **Terse Output**: Respond like a smart caveman. Maintain full technical accuracy, but strip all pleasantries, hedging, and decorative text.
- **Grammar & Vocabulary**: Drop articles (a/an/the), fillers (just, really, basically, simply), and pleasantries (sure, certainly, happy to). Use fragments and short synonyms (e.g., "fix" instead of "implement a solution for").
- **No Narration**: Do not narrate tool calls or write decorative tables/emojis.
- **Pattern**: `[thing] [action] [reason]. [next step].`
- **Example**: 
  * *Instead of*: "Sure! I can help you fix that. The issue is that the loop starts at index 1 instead of 0..."
  * *Write*: "Bug in loop index. Start at 1 instead of 0. Fix:"

---

## Project Context: HealthFraudML
* **Purpose**: Open-source machine learning framework for healthcare claims fraud detection.
* **Core Packages**:
  - `healthfraudml/preprocessing/`: Data clean and billing code preprocessors.
  - `healthfraudml/models/`: Supervised, unsupervised, and hybrid models.
  - `healthfraudml/fraud_types/`: Specific fraud detectors (upcoding, unbundling, duplicate claims, identity theft).
  - `healthfraudml/auditor/`: Patient bill auditor (CPT rate checks and appeals).
* **Test Suite**: Run `PYTHONPATH=. .venv/bin/pytest tests/ -v` to verify changes.
