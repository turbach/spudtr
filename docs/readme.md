Hand edit `./source/index.rst` and `./source/*.rst` beginning with a
capital letter.


To rebuild docs, in a terminal window, navigate to the docs root
directory and send apidoc generated files to `source` like so

```
sphinx-apidoc -f -o source ../spudtr
make html
```
