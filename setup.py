from setuptools import setup, find_packages
import os, sys
import re

# note that to activate the scripts as terminal commands,
# python setup.py develop
# source ~/.(your shell script)

# Use Semantic Versioning, http://semver.org/

# version_info = (0, 4, 0, '')
# __version__ = '%d.%d.%d%s' % version_info


path = os.path.join(os.path.dirname(__file__), "pylibrary")

## Determine current version string
initfile = os.path.join(path, "__init__.py")
init = open(initfile).read()
m = re.search(r"__version__ = (\S+)\n", init)
if m is None or len(m.groups()) != 1:
    raise Exception("Cannot determine __version__ from init file: '%s'!" % initfile)
version = m.group(1).strip("'\"")
initVersion = version

# shamelessly stolen from acq4:
# If this is a git checkout, try to generate a more decriptive version string
try:
    if os.path.isdir(os.path.join(path, ".git")):

        def gitCommit(name):
            commit = check_output(["git", "show", name], universal_newlines=True).split("\n")[0]
            assert commit[:7] == "commit "
            return commit[7:]

        # Find last tag matching "acq4-.*"
        tagNames = check_output(["git", "tag"], universal_newlines=True).strip().split("\n")
        while True:
            if len(tagNames) == 0:
                raise Exception("Could not determine last tagged version.")
            lastTagName = tagNames.pop()
            if re.match(r"acq4-.*", lastTagName):
                break

        # is this commit an unchanged checkout of the last tagged version?
        lastTag = gitCommit(lastTagName)
        head = gitCommit("HEAD")
        if head != lastTag:
            branch = re.search(
                r"\* (.*)", check_output(["git", "branch"], universal_newlines=True)
            ).group(1)
            version = version + "-%s-%s" % (branch, head[:10])

        # any uncommitted modifications?
        modified = False
        status = check_output(["git", "status", "-s"], universal_newlines=True).strip().split("\n")
        for line in status:
            if line.strip() != "" and line[:2] != "??":
                modified = True
                break

        if modified:
            version = version + "+"
    sys.stderr.write("Detected git commit; will use version string: '%s'\n" % version)
except:
    version = initVersion
    sys.stderr.write(
        "This appears to be a git checkout, but an error occurred "
        "while attempting to determine a version string for the "
        "current commit.\nUsing the unmodified version string "
        "instead: '%s'\n" % version
    )
    sys.excepthook(*sys.exc_info())

print("__init__ version: %s  current version: %s" % (initVersion, version))
if "upload" in sys.argv and version != initVersion:
    print("Base version does not match current; stubbornly refusing to upload.")
    exit()

__version__ = version
setup(
    name="pylibrary",
    version=__version__,
    description="General routines to assist in plotting and data analysis",
    url="http://github.com/pbmanis/pylibrary",
    author="Paul B. Manis",
    author_email="pmanis@med.unc.edu",
    license="MIT",
    python_requires=">=3.11",
    install_requires=[
        "matplotlib>=3.5",
        "pyqtgraph>=0.13",
        "numpy>=1.16",
        "scipy>=1.2",
        "cycler>=0.10.0",
        "six>=1.12",
        "pyparsing>=2.4",
        "python-dateutil>=2.8",
        "kiwisolver>=1.1",
        "lmfit>=0.9.3",
    ],
    packages=(
        find_packages() +
        find_packages(where="./pylibrary.tools") +
        find_packages(where="./pylibrary.tools.*") +
        find_packages(where="./pylibrary.plotting") +
        find_packages(where="./pylibrary.plotting.*") +
        find_packages(where="./pylibrary.fitting") +
        find_packages(where="./pylibrary.fitting.*") +
        find_packages(where="./pylibrary.stats") +
        find_packages(where="./pylibrary.stats.*")
    ),
    zip_safe=False,
    entry_points={
        # "console_scripts": [
        #     "listing=pylibrary.tools.listing:main",
        # ],
        "gui_scripts": [
            # no scripts for this library
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11+",
        "Development Status ::  Beta",
        "Environment :: Console",
        "Intended Audience :: Manis Lab",
        "License :: MIT",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Tools :: Python Modules",
        "Topic :: Data Processing :: Neuroscience"
    ],
)
