# paper/

`scripts/make_tables.py` writes `tables_<scale>.tex` here: the LaTeX fragments
from every experiment, concatenated in the order the manuscript uses them, so
the paper can `\input{tables_paper}` a single file.

Put the manuscript sources in this directory. The Docker image mounts it, so a
run inside the container writes the tables straight into the working copy.
