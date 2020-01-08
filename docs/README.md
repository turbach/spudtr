# Documentation generation

To rebuild the complete sphinx html docs, e.g. for a local
preview, in a terminal window navigate to the docs and run

```
make html
```

The docs may be viewed locally by pointing a browser to
`./docs/build/html`


# Files

`./doc/*.rst`

These are the hand edited Sphinx REST docs. The documentation landing
page is the top-level file `./docs/index.rst` and individual sections
are in `./docs/*.rst`


`./docs/source`

The numpy-style (napolean, sphinx_rtd_theme) Reference API docs are
generated automatically in `docs/source` by `api-rst` in the Makefile.
To control the Reference documentation adjust the `sphinx-apidoc`
command parameters.

All files in `source` are auto-generated and overwritten with each doc
build.





