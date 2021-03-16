import pkg_resources


def version():
    return pkg_resources.get_distribution("komono").version


__version__ = version()
