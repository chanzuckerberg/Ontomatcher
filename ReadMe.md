# Project: Ontomatcher


_Ontomatcher_ is a general purpose, configurable, fast mention detector of ontology terms (or entities) in plain text. It is implemented in Python. The detection of mentions relies on text normalization and 'dictionary lookup' of known names and synonyms. The speed comes from its use of the [prefix tree (aka trie)](https://en.wikipedia.org/wiki/Trie) data structure.

## Requirements

The code requires the following packages (versions are those tested):

* Python ver 3.8
* [nltk](https://www.nltk.org), ver 3.5
* [numpy](https://numpy.org), ver 1.19.2
* [pygtrie](https://github.com/google/pygtrie), ver 2.4.2
* [regex](https://github.com/mrabarnett/mrab-regex), ver 2.5.86
* [unidecode](https://pypi.org/project/Unidecode/), ver 1.3.4

### Installing in a Python Virtual Environment

Here is one way to install the Ontomatcher package and all its dependencies, in a Python Virtual Environment. Once you have an appropriate version of Python installed, do the following steps:

1. Create a new virtual environment in `$VENV_BASE/Ontom`, and activate it

	```shell
	$> cd $VENV_BASE
	$> python -m venv --system-site-packages Ontom
	$> . Ontom/bin/activate
	```

2. Clone the Ontomatcher repo into path `$ONTO_BASE/Ontomatcher`

	```shell
	$ ONTO_BASE=...path-to-base-dir...
	$ cd $ONTO_BASE
	$ git clone https://github.com/chanzuckerberg/Ontomatcher.git
	```
	
3. Install Ontomatcher and its dependencies in the virtual environment

	```shell
	$ pip install $ONTO_BASE/Ontomatcher/Python
	```
	
	If not already installed, download nltk's `stopwords`
	
	```shell
	$ python -m nltk.downloader stopwords
	```
	
4. Use it, and then quit the virtual environment

	```shell
	$ python -m ontomatch.nprhub.annotater sample -f $ONTO_BASE/Ontomatcher/Data/plugin_sample.txt $ONTO_BASE/Ontomatcher/Data/imgont_matcher.json
	$ ...
	$ deactivate
	```


## Contents

This repository contains the Ontomatcher code, as well as an application of Ontomatcher for detecting mentions of bio-imaging terms in text, used for recommending categories for [Napari Hub](https://www.napari-hub.org) plugins.

* **Data**: Various data files. Described [here](Data/ReadMe.md).
* **Python**: Python code for the Ontomatcher, and also for the Napari Hub plugin application.

## How To ...

* Develop applications with the Ontomatcher: Start [here](Python/ReadMe.md).
* Use the Napari plugin category recommender: Start [here](Python/ReadMe_NapariPlugin.md).

## Project Status

This project is stable and maintained, but not actively under development.

## License

[MIT](LICENSE)

## Security Issues

[Reporting security issues](SECURITY.md)
