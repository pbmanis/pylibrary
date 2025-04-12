ENVNAME="pylib_venv"
if [ -d $ENVNAME ]
then
    echo "Removing previous environment: $ENVNAME"
    set +e
    # rm -rf $ENVNAME
    #rsync -aR --remove-source-files $ENVNAME ~/.Trash/ || exit 1
    rm -R $ENVNAME
    set -e
else
    echo "No previous environment - ok to proceed"
fi
PYTHONVER="python3.13"
$PYTHONVER -m venv $ENVNAME
source $ENVNAME/bin/activate
pip3 install --upgrade pip  # be sure pip is up to date in the new env.
pip3 install wheel  # seems to be missing (note singular)
$PYTHONVER -m pip install Cython

pip3 install -r requirements.txt
source $ENVNAME/bin/activate
python3 --version
python -m pip install .
# python setup.py develop
