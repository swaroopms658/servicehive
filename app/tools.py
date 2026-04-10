from __future__ import annotations


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock side effect for qualified leads."""
    message = f"Lead captured successfully: {name}, {email}, {platform}"
    print(message)
    return message
