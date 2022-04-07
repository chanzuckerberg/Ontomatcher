# rst-napari/Data/README.md

## Contents

* `EDAM-bioimaging_alpha06.tsv`: The EDAM Bioimaging Ontology version used in this project. Copied from [EDAM Bioimaging Ontology repo](https://github.com/edamontology/edam-bioimaging).

* `curated_imaging_synonyms-220407.csv`: Set of curated synonyms for imaging sub-ontology terms. Format is for reading by the Python function `ontomatch.nprhub.imgontology.read_curated_synonyms()`.

* `imaging_subontology.json`: Describes which subset of the full EDAM ontology will be recognized. See Python function `ontomatch.nprhub.imgontology.get_imaging_subontology()`.

* `imgont_matcher.json`: Options for building an EntityMatcher to recognize mentions of the bio-imaging sub-ontology, used for recommending categories for Napari Hub plug-ins. 

	All paths mentioned in this file are relative to the directory containing this file (i.e. the `Data` dir).

	Used by the following Python functions in the `ontomatch.nprhub.annotater` module:
	* `build_imgont_trie_matcher()`
	*  `get_imgont_trie_matcher()`

* `imgont_stemmed_matcher.json`: Variant of `imgont_matcher.json` that uses word-stemming.

* `plugin_sample.txt`: A sample text file containing a Napari Hub plug-in description, for use as argument to the Python function `ontomatch.nprhub.annotater.test_match_sample()`. The first line contains the name of the plug-in, and the rest of the lines are treated as the description.

* `triematcher_opts.json`: Options for creating an instance of `ontomatch.text.triematcher.TrieMatcher`.

* `triematcher_stemmed_opts.json`: Options for creating an instance of `ontomatch.text.triematcher.TrieMatcher` that uses word stemming.
