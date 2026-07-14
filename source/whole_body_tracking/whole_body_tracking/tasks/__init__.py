"""Package containing task implementations for various robotic environments."""

import importlib
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    package = importlib.import_module(package_name)
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


def _walk_packages(path=None, prefix="", onerror=None, blacklist_pkgs: list[str] | None = None):
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(path_item, memo={}):
        if path_item in memo:
            return True
        memo[path_item] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        if any(black_pkg_name in info.name for black_pkg_name in blacklist_pkgs):
            continue

        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                sub_path = getattr(sys.modules[info.name], "__path__", None) or []
                sub_path = [path_item for path_item in sub_path if not seen(path_item)]
                yield from _walk_packages(sub_path, info.name + ".", onerror, blacklist_pkgs)

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
