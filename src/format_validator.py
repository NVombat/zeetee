import os
import json
import jsonschema
from jsonschema import Draft7Validator

from . import *
from .logger import create_logger

logger = create_logger(l_name="zt_fmt_validator")


instance_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "patternProperties": {
        "^[0-9]+$": {
            "type": "object",
            "properties": {
                "i": {
                    "type": "integer",
                    "minimum": 0
                },
                "n": {
                    "type": "integer",
                    "minimum": 0
                },
                "t": {
                    "type": "integer",
                    "minimum": 0
                },
                "partitions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "integer",
                            "minimum": 1
                        }
                    }
                },
                "flag": {
                    "type": "integer",
                    "enum": [0, 1]
                },
                "uc": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "integer",
                                    "minimum": 1
                                }
                            },
                            {
                                "type": "string",
                                "enum": ["le"]
                            },
                            {
                                "type": "integer",
                                "minimum": 0
                            }
                        ],
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "sc": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "integer",
                                    "minimum": 1
                                }
                            },
                            {
                                "type": "string",
                                "enum": ["ge"]
                            },
                            {
                                "type": "integer",
                                "minimum": 0
                            }
                        ],
                        "minItems": 3,
                        "maxItems": 3
                    }
                }
            },
            "required": ["i", "n", "t", "partitions", "flag", "uc", "sc"],
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}


def validate_json_format(json_file_path=default_jfp) -> bool:
    '''
    Validates format of RGP Instances stored in the JSON file

    Args:
        json_file_path: Path to JSON file

    Returns:
        bool: True/False based on format
    '''
    with open(json_file_path, "r") as fh:
        rgp_json_inst = json.load(fh)

    validator = Draft7Validator(instance_schema)

    # Collect all validation errors
    errors = sorted(validator.iter_errors(rgp_json_inst), key=lambda e: e.path)

    if errors:
        # Log each error
        for error in errors:
            logger.error(f"{error.message} at path {'/'.join(map(str, error.path))}")

        return False

    else:
        logger.debug(f"JSON FILE {json_file_path} is VALID")
        return True


if __name__ == "__main__":
    logger.info("********************FORMAT_VALIDATOR[LOCAL_TESTING]*********************")

    fmt = validate_json_format()
    # fmt = validate_json_format(neg_jfp)
    # fmt = validate_json_format(small_jfp)
    # fmt = validate_json_format(multi_jfp)
    # fmt = validate_json_format(test_path_neg)
    # fmt = validate_json_format(false_fmt_jfp)

    if fmt:
        logger.info("RGP Instances Have VALID Format!")
    else:
        logger.error(f"RGP Instances Have INVALID Format!")