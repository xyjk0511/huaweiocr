import importlib
import os
import sys
import threading
import types
import unittest
from unittest import mock


def _import_ocr_with_stubs():
    for name in ("ocr", "huaweiocr.detect.ocr_engine", "paddle", "paddleocr", "app_paths"):
        sys.modules.pop(name, None)
    paddle = types.ModuleType("paddle")
    paddle.set_device = lambda _device: None
    paddleocr = types.ModuleType("paddleocr")
    paddleocr.PaddleOCR = type("DummyPaddleOCR", (), {})
    app_paths = types.ModuleType("app_paths")
    app_paths.ensure_models_installed = lambda: None
    app_paths.get_resource_path = lambda *parts: os.path.join(*parts)
    sys.modules.update({"paddle": paddle, "paddleocr": paddleocr, "app_paths": app_paths})
    return importlib.import_module("ocr")


class LangchainImportCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr = _import_ocr_with_stubs()

    def _fake_langchain(self, **submodule_specs):
        langchain = types.ModuleType("langchain")
        langchain.__path__ = []
        real_find_spec = importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            if name == "langchain":
                return object()
            if name in submodule_specs:
                return submodule_specs[name]
            if name.startswith("langchain."):
                return None
            return real_find_spec(name, *args, **kwargs)

        modules = {name: None for name in list(sys.modules) if name.startswith("langchain.")}
        modules["langchain"] = langchain
        return langchain, modules, find_spec

    def test_slots_are_lazy_and_resolve_to_real_classes(self):
        langchain, modules, find_spec = self._fake_langchain()
        splitter_mod = types.ModuleType("langchain_text_splitters")
        splitter_mod.RecursiveCharacterTextSplitter = type(
            "RealSplitter", (), {"__init__": lambda self, chunk_size=1: setattr(self, "size", chunk_size)}
        )
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(importlib.util, "find_spec", find_spec):
            for name in [n for n, v in modules.items() if v is None]:
                sys.modules.pop(name, None)
            self.ocr.ensure_langchain_import_compat()
            self.assertNotIn("langchain_text_splitters", sys.modules)
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.docstore.document import Document

            self.assertTrue(callable(Document))
            with mock.patch.dict(sys.modules, {"langchain_text_splitters": splitter_mod}):
                splitter = RecursiveCharacterTextSplitter(chunk_size=7)
        self.assertEqual(type(splitter).__name__, "RealSplitter")
        self.assertEqual(splitter.size, 7)

    def test_stand_ins_support_classmethods_isinstance_and_swap_in_real_class(self):
        langchain, modules, find_spec = self._fake_langchain()
        core = types.ModuleType("langchain_core.documents")

        class RealDocument:
            def __init__(self, page_content):
                self.page_content = page_content

            @classmethod
            def build(cls, text):
                return cls(text)

        core.Document = RealDocument
        with mock.patch.dict(sys.modules, modules),                 mock.patch.object(importlib.util, "find_spec", find_spec):
            for name in [n for n, v in modules.items() if v is None]:
                sys.modules.pop(name, None)
            self.ocr.ensure_langchain_import_compat()
            from langchain.docstore.document import Document

            self.assertEqual(sys.modules["langchain.docstore"].__spec__.name, "langchain.docstore")
            with mock.patch.dict(sys.modules, {"langchain_core.documents": core}):
                doc = Document.build("AP162E")
                self.assertIsInstance(doc, Document)
                self.assertIs(sys.modules["langchain.docstore.document"].Document, RealDocument)
        self.assertEqual(doc.page_content, "AP162E")

    def test_broken_langchain_install_does_not_block_ocr_import(self):
        real_find_spec = importlib.util.find_spec
        with mock.patch.object(
            importlib.util, "find_spec",
            lambda name, *a, **k: object() if name == "langchain" else real_find_spec(name, *a, **k),
        ), mock.patch.dict(sys.modules, {"langchain": None}):
            self.ocr.ensure_langchain_import_compat()

    def test_real_parent_module_skips_only_its_children(self):
        langchain, modules, find_spec = self._fake_langchain(**{"langchain.docstore": object()})
        with mock.patch.dict(sys.modules, modules),                 mock.patch.object(importlib.util, "find_spec", find_spec):
            for name in [n for n, v in modules.items() if v is None]:
                sys.modules.pop(name, None)
            self.ocr.ensure_langchain_import_compat()
            self.assertFalse(hasattr(langchain, "docstore"))
            self.assertTrue(hasattr(sys.modules["langchain.text_splitter"], "RecursiveCharacterTextSplitter"))

    def test_existing_real_submodule_is_not_shadowed(self):
        langchain, modules, find_spec = self._fake_langchain(**{"langchain.text_splitter": object()})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(importlib.util, "find_spec", find_spec):
            for name in [n for n, v in modules.items() if v is None]:
                sys.modules.pop(name, None)
            self.ocr.ensure_langchain_import_compat()
            self.assertFalse(hasattr(langchain, "text_splitter"))
            self.assertNotIn("langchain.text_splitter", sys.modules)

    def test_noop_without_langchain(self):
        real_find_spec = importlib.util.find_spec
        with mock.patch.object(
            importlib.util, "find_spec",
            lambda name, *a, **k: None if name == "langchain" else real_find_spec(name, *a, **k),
        ), mock.patch.dict(sys.modules, {}):
            sys.modules.pop("langchain", None)
            self.ocr.ensure_langchain_import_compat()
            self.assertNotIn("langchain", sys.modules)


class ConcurrentOcrModuleLoadTest(unittest.TestCase):
    def test_concurrent_first_load_imports_once_and_returns_complete_module(self):
        from huaweiocr.recognition import pipeline

        imports = []
        gate = threading.Event()

        class SlowOcrFinder:
            def find_spec(self, name, path=None, target=None):
                if name != "ocr":
                    return None
                return importlib.util.spec_from_loader(name, self)

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                imports.append(module)
                gate.wait(2)
                module.ocr_one_image = lambda *_a: "ok"

        results = []
        finder = SlowOcrFinder()
        with mock.patch.object(pipeline, "OCR_MODULE", None), \
                mock.patch.dict(sys.modules, {}), \
                mock.patch.object(sys, "meta_path", [finder] + sys.meta_path):
            sys.modules.pop("ocr", None)
            threads = [
                threading.Thread(target=lambda: results.append(pipeline._load_ocr_module()))
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            gate.set()
            for thread in threads:
                thread.join(5)
        self.assertEqual(len(imports), 1)
        self.assertEqual(len(results), 4)
        for module in results:
            self.assertEqual(module.ocr_one_image(), "ok")


class CliOcrPrewarmTest(unittest.TestCase):
    def _scan2(self, fallback=True):
        started = threading.Event()
        scan2 = types.SimpleNamespace(
            scan_ocr_fallback_enabled=lambda: fallback,
            prewarm_ocr_engine=started.set,
        )
        return scan2, started

    def test_cli_prewarms_ocr_in_background_by_default(self):
        from huaweiocr import cli

        scan2, started = self._scan2()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HUAWEIOCR_PREWARM_OCR", None)
            thread = cli._start_ocr_prewarm(scan2)
        self.assertIsNotNone(thread)
        thread.join(5)
        self.assertTrue(started.is_set())
        self.assertTrue(thread.daemon)

    def test_cli_prewarm_respects_opt_out_and_disabled_fallback(self):
        from huaweiocr import cli

        scan2, started = self._scan2()
        with mock.patch.dict(os.environ, {"HUAWEIOCR_PREWARM_OCR": "0"}):
            self.assertIsNone(cli._start_ocr_prewarm(scan2))
        scan2, started = self._scan2(fallback=False)
        self.assertIsNone(cli._start_ocr_prewarm(scan2))
        scan2, started = self._scan2()
        scan2.OCR_THREAD_LOCAL_MODE = True
        self.assertIsNone(cli._start_ocr_prewarm(scan2))
        self.assertFalse(started.is_set())

    def test_gui_prewarm_defaults_on_like_scan2_fallback(self):
        import gui_pipeline

        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(threading.Thread, "start", lambda self: None):
            os.environ.pop("HUAWEIOCR_PREWARM_OCR", None)
            os.environ.pop("SCAN2_OCR_FALLBACK", None)
            thread = gui_pipeline.start_ocr_prewarm_thread()
        self.assertIsNotNone(thread)


if __name__ == "__main__":
    unittest.main()
