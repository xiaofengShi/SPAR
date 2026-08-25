import importlib
import unittest


class CoreImportSmokeTests(unittest.TestCase):
    def test_public_entrypoint_modules_import(self):
        for module_name in (
            "demo_app_with_front",
            "pipeline_spar",
            "search_engine",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
