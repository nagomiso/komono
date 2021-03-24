import pkg_resources


def version() -> str:
    return pkg_resources.get_distribution("komono").version


__version__ = version()
