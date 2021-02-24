# MLFlow Operationalisation Article

Here we have the main essay (in article format) that relates our accomplishments, decisions and 
whatever we have faced during this EngLab initiative. This essay is written as a markdown file but 
one can use [pandoc](http://pandoc.org/) to convert to other formats.

Important assets in the folder:

- `article.md`: Essay written in markdown;
- `Dockefile-pandoc`: Dockerfile that contains pandoc and other utilities to process the essay;
- `gh.html5`: GitHub-based template used by pandoc to generate html files. This template is 
    extracted from this [repository](https://github.com/jgm/pandoc-templates);
- `references.bib`: All references used in the essay;
- `ieee-url-csl`: Citation and Reference style extracted from 
    [Zotero Repository](https://www.zotero.org/styles);
- `png files`: Local figures used in the essay.

## Using Pandoc

You must install `pandoc`, `pandoc-crossref`and `pandoc-citeproc` before processing the markdown 
file. After that, this command line should fit:

```sh
    $ pandoc --bibliography references.bib -F pandoc-crossref --citeproc --csl iee-url.csl \
      --template gh.html5 article.md -o article.html
```


In case you don't have (or don't want to install) pandoc, we provide a 
[Dockerfile](Dockerfile-pandoc) ready to be used. The commands bellow output the same generate html
file you would get if you have pandoc installed locally.

```sh
    $ docker build -t englabs/pandoc -f Dockerfile-pandoc .
    $ docker run --rm -v <path/to/article/folder>:/home/pandoc/article englabs/pandoc:latest
```

**Notes:**
- Pandoc builds a tree to represent the document and after that it may use filters (`-F`) to process
  the representation (`pandoc-crossref`and `pandoc-citeproc`are filters that parse information and
  generate references, citations and labels);
- It infers the output format from the file output extension. You may use `-t` to set a target 
  format (if you do that, be sure to have proper styles and templates);
- You may change the command executed in docker container. Just add the arguments after the image 
  name.

