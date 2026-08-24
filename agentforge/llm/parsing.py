"""Strict JSON extraction; intentionally never executes model output."""

import json
import re
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str):
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char in "[{":
            try:
                value, end = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if cleaned[index + end:].strip():
                raise ValueError("unexpected text after JSON value")
            return value
    raise ValueError("response does not contain valid JSON")


def parse_model(text: str, schema: type[T]) -> T:
    return schema.model_validate(extract_json(text))


def strip_python_fence(text: str) -> str:
    value = text.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", value, flags=re.I | re.S)
    return match.group(1).strip() + "\n" if match else value + "\n"
