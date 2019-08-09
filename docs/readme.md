Files in `./source/*.rst` beginning with a capital letter are edited
by hand.

To rebuild docs, in a terminal window, navigate to the docs root
directory and send apidoc generated files to `source` like so

```
sphinx-apidoc -f -o source ../pfunky
make html
```
