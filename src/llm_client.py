"""LLM Tracking and API Client Module using google-genai and python-dotenv."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

# Find repository root directory to ensure .env is correctly loaded
REPO_ROOT = Path(__file__).resolve().parents[1]


class LLMTracker:
    """Tracks Gemini LLM API calls, token counts, and costs across requests."""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv(dotenv_path=REPO_ROOT / ".env")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key or self.api_key == "your-key-here":
            logger.warning(
                "GEMINI_API_KEY is missing or contains default placeholder in .env."
            )

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.model_usage: Dict[str, Dict[str, int]] = {}

    def ask_gemini(
        self,
        prompt: Any,
        model_name: str = "gemini-3.6-flash",
        config: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Calls the Gemini API with prompt (text or list of parts/images), tracks input/output tokens, and updates running totals.

        Args:
            prompt: Text prompt or list of content parts (e.g. [image_part, text_prompt]) to send.
            model_name: Name of the Gemini model to use (default: gemini-3.6-flash).
            config: Optional configuration object or dict (e.g. types.GenerateContentConfig).

        Returns:
            str: Generated text response from Gemini API.
        """
        if not self.client:
            raise ValueError(
                "Gemini client is not initialized. Please set a valid GEMINI_API_KEY in .env."
            )

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                kwargs = {"model": model_name, "contents": prompt}
                if config is not None:
                    kwargs["config"] = config
                response = self.client.models.generate_content(**kwargs)
                self.total_calls += 1

                # Extract token counts safely from usage_metadata
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                # Update per-model breakdown
                if model_name not in self.model_usage:
                    self.model_usage[model_name] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                self.model_usage[model_name]["calls"] += 1
                self.model_usage[model_name]["input_tokens"] += input_tokens
                self.model_usage[model_name]["output_tokens"] += output_tokens

                logger.info(
                    f"API Call #{self.total_calls} [{model_name}]: "
                    f"Prompt Tokens={input_tokens}, Output Tokens={output_tokens}"
                )
                return response.text

            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                    sleep_time = 2
                    logger.warning(
                        f"Rate limit 429 hit for {model_name}. Retrying in {sleep_time}s (Attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    continue

                logger.error(f"Failed Gemini API call to {model_name}: {e}")
                raise

    def export_usage_report(
        self, filepath: str = "evaluation/usage_report.md"
    ) -> Path:
        """
        Writes a markdown usage report summarizing API calls, token counts, and cost estimates.

        Args:
            filepath: Destination file path for the usage report markdown.

        Returns:
            Path: Path object pointing to the written markdown file.
        """
        target_path = (
            REPO_ROOT / filepath if not Path(filepath).is_absolute() else Path(filepath)
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        avg_input = (
            self.total_input_tokens / self.total_calls if self.total_calls > 0 else 0.0
        )
        avg_output = (
            self.total_output_tokens / self.total_calls if self.total_calls > 0 else 0.0
        )
        total_tokens = self.total_input_tokens + self.total_output_tokens

        # Estimated cost calculation (gemini-2.5-flash rates placeholder)
        est_input_cost = (self.total_input_tokens / 1_000_000) * 0.075
        est_output_cost = (self.total_output_tokens / 1_000_000) * 0.30
        est_total_cost = est_input_cost + est_output_cost
        avg_cost_per_call = (
            est_total_cost / self.total_calls if self.total_calls > 0 else 0.0
        )

        report_content = f"""# LLM Token & API Usage Report

## Summary Statistics
- **Total API Calls:** {self.total_calls}
- **Total Input (Prompt) Tokens:** {self.total_input_tokens:,}
- **Total Output (Candidates) Tokens:** {self.total_output_tokens:,}
- **Total Tokens Used:** {total_tokens:,}
- **Average Input Tokens / Call:** {avg_input:.2f}
- **Average Output Tokens / Call:** {avg_output:.2f}

## Cost Estimation (Placeholder)
- **Estimated Input Cost:** ${est_input_cost:.6f} (@ $0.075 / 1M tokens)
- **Estimated Output Cost:** ${est_output_cost:.6f} (@ $0.30 / 1M tokens)
- **Estimated Total Cost:** ${est_total_cost:.6f}
- **Estimated Cost per Request:** ${avg_cost_per_call:.6f}

## Model Breakdown
"""
        if self.model_usage:
            for m_name, stats in self.model_usage.items():
                report_content += (
                    f"- **{m_name}**: {stats['calls']} calls, "
                    f"{stats['input_tokens']:,} input tokens, "
                    f"{stats['output_tokens']:,} output tokens\n"
                )
        else:
            report_content += "- No successful API calls recorded yet.\n"

        target_path.write_text(report_content, encoding="utf-8")
        logger.info(f"Usage report exported to {target_path}")
        return target_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = LLMTracker()
    try:
        print("Sending test prompt to Gemini...")
        response_text = tracker.ask_gemini("What is 2 + 2? Answer in one sentence.")
        print(f"Gemini Response: {response_text}")
    except Exception as err:
        print(f"API Call failed (expected if API key is invalid/placeholder): {err}")

    report_file = tracker.export_usage_report("evaluation/usage_report.md")
    print(f"Report generated at: {report_file}")
