from setuptools import setup, find_packages
import os

# note that to activate the scripts as terminal commands, 
# python setup.py develop
# source ~/.(your shell script)

# Use Semantic Versioning, http://semver.org/

# version_info = (0, 4, 0, '')
# __version__ = '%d.%d.%d%s' % version_info


path = os.path.join(os.path.dirname(__file__), 'pylibrary')
version = None
for line in open(os.path.join(path, '__init__.py'), 'r').readlines():
    if line.startswith('__version__'):
        version = line.partition('=')[2].strip('"\' \n')
        break
if version is None:
    raise Exception("Could not read __version__ from pylibrary/__init__.py")

__version__ =  version
setup(name='pylibrary',
      version=__version__,
      description='General routines to assist in plotting and data analysis',
      url='http://github.com/pbmanis/pylibrary',
      author='Paul B. Manis',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['matplotlib>=3.5', 'pyqtgraph>=0.13',
          'numpy>=1.16', 'scipy>=1.2', 'cycler>=0.10.0', 'six>=1.12',
          'pyparsing>=2.4', 'python-dateutil>=2.8', 'kiwisolver>=1.1',
          'lmfit>=0.9.3',
          ],
      packages=find_packages(include=['pylibrary']),
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'listing=pylibrary.tools.listing:main',
              ],
          'gui_scripts': [
               # no scripts for this library
          ],
      },
      classifiers = [
             "Programming Language :: Python :: 3.6+",
             "Development Status ::  Beta",
             "Environment :: Console",
             "Intended Audience :: Manis Lab",
             "License :: MIT",
             "Operating System :: OS Independent",
             "Topic :: Software Development :: Tools :: Python Modules",
             "Topic :: Data Processing :: Neuroscience",
             ],
    )
