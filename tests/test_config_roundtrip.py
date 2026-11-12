"""Ensure saved experiment settings retain every dataclass field."""
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import Config


def change_every_field(instance):
    for field in fields(instance):
        value = getattr(instance, field.name)
        if is_dataclass(value):
            change_every_field(value)
        elif isinstance(value, bool):
            setattr(instance, field.name, not value)
        elif isinstance(value, int):
            setattr(instance, field.name, value + 17)
        elif isinstance(value, float):
            setattr(instance, field.name, value + 0.125)
        elif isinstance(value, str):
            setattr(instance, field.name, value + "_roundtrip")
        elif isinstance(value, list):
            setattr(instance, field.name, list(value) + [11])
        else:
            raise AssertionError(f"Add a non-default fixture for new field: {field.name}")


class ConfigRoundTripTests(unittest.TestCase):
    def test_to_dict_includes_every_field(self):
        config = Config()
        change_every_field(config)
        self.assertEqual(config.to_dict(), asdict(config))

    def test_save_load_retains_all_nondefault_values(self):
        config = Config()
        change_every_field(config)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "config.json")
            config.save(path)
            loaded = Config.load(path)
        self.assertEqual(asdict(config), asdict(loaded))


if __name__ == "__main__":
    unittest.main()
