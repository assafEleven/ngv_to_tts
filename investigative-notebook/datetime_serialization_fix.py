#!/usr/bin/env python3
"""
Datetime Serialization Fix - ASCII-only for Vertex AI compatibility
Handles datetime objects in JSON serialization with production-ready error handling
"""

import json
import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects and other common types"""
    
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, datetime.time):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return super().default(obj)

def deep_serialize_datetime(obj: Any) -> Any:
    """
    Deep serialization of datetime objects in nested structures
    Returns JSON-serializable objects with ASCII-only output
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, datetime.time):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: deep_serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deep_serialize_datetime(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_serialize_datetime(item) for item in obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safe JSON serialization with datetime handling
    Uses ASCII-only output for Vertex AI compatibility
    """
    try:
        # First try with custom encoder
        return json.dumps(obj, cls=DateTimeEncoder, ensure_ascii=True, **kwargs)
    except (TypeError, ValueError) as e:
        # Fallback: deep serialize then dump
        try:
            serialized_obj = deep_serialize_datetime(obj)
            return json.dumps(serialized_obj, ensure_ascii=True, **kwargs)
        except (TypeError, ValueError) as e2:
            # Final fallback: convert everything to string
            return json.dumps(str(obj), ensure_ascii=True, **kwargs)

def serialize_agent_result(obj: Any) -> Any:
    """
    Serialize agent results with proper datetime handling
    Specifically designed for AgentResult objects
    """
    if hasattr(obj, '__dict__'):
        # Handle dataclass or object with attributes
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = deep_serialize_datetime(value)
        return result
    else:
        return deep_serialize_datetime(obj)

def serialize_investigation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize investigation data with comprehensive datetime handling
    """
    return deep_serialize_datetime(data)

def safe_timestamp_format(timestamp: Union[datetime.datetime, str, None]) -> str:
    """
    Format timestamp safely for display
    Returns ISO format string or fallback
    """
    if timestamp is None:
        return "N/A"
    elif isinstance(timestamp, datetime.datetime):
        return timestamp.isoformat()
    elif isinstance(timestamp, str):
        return timestamp
    else:
        return str(timestamp)

# Example usage and test functions
def test_datetime_serialization():
    """Test datetime serialization functionality"""
    print("Testing datetime serialization...")
    
    # Test data with various datetime objects
    test_data = {
        "timestamp": datetime.datetime.now(),
        "date": datetime.date.today(),
        "time": datetime.time(14, 30, 45),
        "nested": {
            "created_at": datetime.datetime.now(),
            "list_with_dates": [datetime.datetime.now(), datetime.date.today()]
        },
        "regular_data": "test string",
        "number": 42
    }
    
    # Test serialization
    try:
        serialized = deep_serialize_datetime(test_data)
        json_str = safe_json_dumps(serialized)
        print("SUCCESS: Datetime serialization test passed")
        print(f"Sample output: {json_str[:100]}...")
        return True
    except Exception as e:
        print(f"ERROR: Datetime serialization test failed: {e}")
        return False

if __name__ == "__main__":
    test_datetime_serialization() 