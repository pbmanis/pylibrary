from setuptools import setup, find_packages

import os


path = os.path.join(os.path.dirname(__file__), 'pylibrary')
version = None
for line in open(os.path.join(path, '__init__.py'), 'r').readlines():
    if line.startswith('__version__'):
        version = line.partition('=')[2].strip('"\' \n')
        break
if version is None:
    raise Exception("Could not read __version__ from pylibrary/__init__.py")


setup(name='pylibrary',
      version=version,
      description='Paul Manis, Python Utility Library',
      url='https://',
      author='Paul B. Manis, Ph.D.',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['pylibrary*', 'pylibrary.xlsx']),
      zip_safe=False)
      