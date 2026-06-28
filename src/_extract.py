import sys, importlib
mod = importlib.import_module(sys.argv[1])
lesson = getattr(mod, sys.argv[2])
lang = sys.argv[3] if len(sys.argv) > 3 else 'en'
print(lesson[lang])
