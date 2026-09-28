"""
Lab 11 — Helper Utilities
"""
import asyncio

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    try:
        async with asyncio.timeout(90):
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
    except asyncio.TimeoutError:
        final_response = "[ERROR: LLM call timed out after 90s]"
    except Exception as exc:
        final_response = f"[ERROR: {type(exc).__name__}: {exc}]"

    # Robust fallback: If ADK runner failed (e.g. 503 spike, tenacity deadlock, or 429),
    # call google.genai directly using the agent's system instruction.
    if not final_response or final_response.startswith("[ERROR:"):
        try:
            import os
            key = os.environ.get("GOOGLE_API_KEY", "")
            instruction = getattr(agent, "instruction", None)
            if key and instruction:
                from google import genai
                from google.genai import types

                model = getattr(agent, "model", None) or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
                client = genai.Client(api_key=key)
                for try_model in [model, "gemini-2.5-flash"]:
                    try:
                        res = client.models.generate_content(
                            model=try_model,
                            contents=user_message,
                            config=types.GenerateContentConfig(
                                system_instruction=instruction,
                                temperature=0.7,
                                max_output_tokens=512,
                            ),
                        )
                        if res.text:
                            final_response = res.text
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    return final_response, session
