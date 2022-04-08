# Making Category Recommendations for Napari Plugins
<hr>

## Requirements

For code requirements, go [here](../ReadMe.md#requirements).

## Code

The application code for Napari Hub plugin category recommendations is all contained within the [ontomatch.nprhub](ontomatch/nprhub) package. All the data files can be found in the [Data directory](../Data).


## Command Line Operations

For convenience, all the following commands assume the user is in the `Ontomatcher/Python` directory, and the `Ontomatcher/Data` directory is a peer directory.

### Build an Instance of the Bioimaging Terms Mention Detector

Use the following command to build an instance of `ontomatch.text.triematcher.TrieMatcher` populated for detecting Napari Hub plugin category terms.

```
$> python -m ontomatch.nprhub.annotater build ../Data/imgont_matcher.json
```

This will create an instance of the curated sub-ontology of EDAM, create an instance of the `TrieMatcher` with configuration specified in `../Data/triematcher_opts.json`, and then populate it with the curated ontology terms and synonyms. It then caches the `TrieMatcher` as a pickle file at the path specified in `imgont_matcher.json`.

**NOTE**: If you change the options since the last build, but want to use the same cache file, first delete the old pickle file, and then invoke the above command.


### Output Various Statistics for the Mention Detector

```
$> python -m ontomatch.nprhub.annotater stats ../Data/imgont_matcher.json
```

This command outputs to `stdout` various ontology and mention detector statistics.


### Detect Bio-imaging Term Mentions

To detect mentions of terms (aka entities) from the curated imaging sub-ontology, for text stored in a file (in this example `../Data/plugin_sample.txt`), use the following command:

```
$> python -m ontomatch.nprhub.annotater sample -f ../Data/plugin_sample.txt ../Data/imgont_matcher.json
```

Here `../Data/imgont_matcher.json` specifies the instance of the mention detector to use.

To detect mentions in an actual plugin, specify the name of the plugin, as follows (the example uses `napari-allencell-segmenter`):

```
$> python -m ontomatch.nprhub.annotater sample -n napari-allencell-segmenter ../Data/imgont_matcher.json
```


## Python API

### Bio-imaging Ontology

We use a subset of the [EDAM Bioimaging Ontology](https://github.com/edamontology/edam-bioimaging) as categories for Napari Hub plugins. This subset is specified as follows:

* The version of the EDAM ontology: [EDAM-bioimaging_alpha06](../Data/EDAM-bioimaging_alpha06.tsv)
* The relevant sub-trees in the EDAM ontology: expressed in [this JSON file](../Data/imaging_subontology.json)
* A set of curated synonyms for each term: specified in [this file](../Data/curated_imaging_synonyms-220407.csv).

The Python function `ontomatch.nprhub.imgontology.get_curated_imaging_subontology()` can be used to create an instance of the `ontomatch.data.ontology.Ontology` class that contains the relevant curated sub-ontology.


### Building a Mention Detector

The wrapper function `ontomatch.nprhub.annotater.build_imgont_trie_matcher()` will build an instance of the class `ontomatch.text.triematcher.TrieMatcher` with specified options, and populate it with curated terms.

The function `ontomatch.nprhub.annotater.get_imgont_trie_matcher()` will retrieve a pre-built instance of the mention detector, and also build it if a pre-built instance does not exist.

#### Example TrieMatcher Options

Both the above functions expect a Python dictionary specifying various configurable options, which can also be provided as a path to a JSON file. The file used in the examples here is [../Data/triematcher_opts.json](../Data/triematcher_opts.json):

```
{
    "class": "TrieMatcher",
    "name": "TrieMatcher - Unstemmed",
    "descr": "Trie Matcher, 3 tiers, unstemmed.",
    "lexicon_id": "EDAM Imaging sub-Ontology",
    "name_types": {
        "primary": {
            "tier": 1,
            "normalization": "LOWER"
        },
        "acronym": {
            "tier": 2,
            "normalization": "STANDARD"
        },
        "synonym": {
            "tier": 1,
            "normalization": "LOWER"
        },
        "partial": {
            "tier": 4,
            "normalization": "LOWER"
        }
    },
    "cache_file": "imgont_matcher.pkl"
}
```

Detailed description of the options:

* `"class": "TrieMatcher"`: Required entry specifying the name of the class that will be instantiated.
* `"name": "TrieMatcher - Unstemmed"`: A convenience name for this instance.
* `"descr": "Trie Matcher, 3 tiers, unstemmed."`: A more detailed description of this instance, for documentation.
* `"lexicon_id": "EDAM Imaging sub-Ontology"`: Documents which ontology is used. The function `build_imgont_trie_matcher()` populates this with the path to the curated synonyms file.
* `"name_types"`: A required entry, specifies a dictionary mapping various named Name-Types to their parameters. Each name-type specifies the `tier` (matches with lower tier numbers are preferred), and how the text and synonyms are standardized before looking for matches (`normalization`).
* `"primary"`: This entry is required, and specifies how the Primary Name of an Entity or Term is matched. In this example it is assigned `tier 1` (the most important tier), and `normalization` value `LOWER` (text is tokenized, and then converted to lower-case; term names are also processed the same way).
* `"acronym"`: These are matched in a case-sensitive manner after tokenization (`"normalization": "STANDARD"`) and matches are assigned `tier 2`, which is one 'lower' than those for Primary Names.
* `"synonym"`: All full synonyms for the term are handled exactly like its Primary Name.
* `"partial"`: Is the name type used for 'partial synonyms' (e.g. "3D" may be a partial synonym for "3D Image"). These are given a 'lower' tier than Acronyms.
* `"cache_file"`: This specifies the location of the file where the built instance of the matcher will be saved, as a pickle file. The path here is relative to the directory containing the options file (e.g. relative to the `Ontomatcher/Data` directory when the options file path is `Ontomatcher/Data/triematcher_opts.json`). In this example, the built instance will be cached to `Ontomatcher/Data/imgont_matcher.pkl`.


### Detect Imaging Term Matches in Text

The basic function for finding all non-overlapping mentions in plain `text` is `TrieMatcher.get_greedy_nonoverlapping_matches(text)`. There are several convenience wrapper functions provided in the `ontomatch.nprhub.annotater` module:

* `get_ont_matches_by_line(text)` will first split `text` into lines, based on the assumption that mentions will not cross line boundaries, and then look for mentions in each line. The output is the list of text lines, and for each line, a list of matches. Each match specifies the Term ID (`entity_id`) of the matching term, and the exact location in the corresponding text-line where that term's mention was detected.

	An alternative is to split text into sentences, e.g. using [nltk's sentence splitter](https://www.nltk.org/api/nltk.tokenize.html).

* `get_entity_matches(text)`: This is a wrapper around `get_ont_matches_by_line(text)`, and returns an `EntityMatch` object for each matching term (or entity), which includes the actual synonyms responsible for the detected match.

* `get_matching_entity_ids(text)`: Another convenience wrapper which simply returns the set of matching entity-IDs (aka term-IDs).

* `pp_ont_matches()`: This is the function behind the command line:

	```
	$> python -m ontomatch.nprhub.annotater sample ...
	```
	Look at the implementation to see how term matches are mapped to locations in the source text, and also how matching terms are divided among the broad categories used by Napari Hub.