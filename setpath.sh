#
# This script adds Mitsuba to the current path.
# It works with both Bash and Zsh.
#
# NOTE: this script must be sourced and not run, i.e.
#    . setpath.sh        for Bash
#    source setpath.sh   for Zsh or Bash
#

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "The setpath.sh script must be sourced, not executed. In other words, run\n"
	echo "$ source setpath.sh\n"
	echo "If you wish to use the Mitsuba Python bindings, you should also specify"
	echo "your Python version /before/ sourcing setpath.sh, e.g.\n"
	echo "$ export PSDRPHY_VER=3.3"
	echo "$ source setpath.sh"
	exit 0
fi

if [ "$BASH_VERSION" ]; then
	PSDR_DIR=$(dirname "$BASH_SOURCE")
	export PSDR_DIR=$(builtin cd "$PSDR_DIR"; builtin pwd)
elif [ "$ZSH_VERSION" ]; then
	export PSDR_DIR=$(dirname "$0:A")
fi

if [ "$PSDRPHY_VER" ]; then
	pyver=$PSDRPHY_VER
else
	pyver=`python --version 2>&1 | grep -oE '([[:digit:]].[[:digit:]])' | head -n1`
fi

if [[ "$(uname)" == 'Darwin' ]]; then
	export PYTHONPATH="$PSDR_DIR/build/lib:$PSDR_DIR/ext/enoki/build:$PYTHONPATH"
else
	export PYTHONPATH="$PSDR_DIR/build/lib:$PSDR_DIR/ext/enoki/build:$PYTHONPATH"
fi
unset pyver

if [[ "$(uname)" == 'Darwin' ]]; then
	export PATH="$PSDR_DIR/build:$PSDR_DIR/ext/enoki/build:$PATH"
else
	export LD_LIBRARY_PATH="$PSDR_DIR/build/lib:$PSDR_DIR/ext/enoki/build:$LD_LIBRARY_PATH"
	export PATH="$PSDR_DIR/build:$PSDR_DIR/ext/enoki/build:$PATH"
fi
