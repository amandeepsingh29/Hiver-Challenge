import json
import pytest
from generator import GeneratorOutput

def test_generator_schema_validation():
    # Test valid JSON
    valid_json = '{"intent": "refund", "requires_human": false, "suggested_reply": "Here is your refund."}'
    output = GeneratorOutput.model_validate_json(valid_json)
    
    assert output.intent == "refund"
    assert output.requires_human is False
    assert "refund" in output.suggested_reply

def test_generator_schema_invalid():
    # Test invalid JSON (missing required fields)
    invalid_json = '{"intent": "refund"}'
    with pytest.raises(Exception):
        GeneratorOutput.model_validate_json(invalid_json)
