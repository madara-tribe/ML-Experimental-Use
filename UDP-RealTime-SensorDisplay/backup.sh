#!/bin/sh
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name '.DS_Store' -type f -ls -delete
mkdir -p bp bp/qtWidgets bp/matplotlib/mpl-data
cp *.txt *.py *.sh bp/
cp qtWidgets/*.py bp/qtWidgets/
cp matplotlib/mpl-data/* bp/matplotlib/mpl-data/
