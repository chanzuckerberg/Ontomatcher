# rst-napari/Data/README.md

## Contents

* `EDAM-bioimaging_alpha06.tsv`: The EDAM Bioimaging Ontology version used in this project. Copied from [EDAM Bioimaging Ontology repo](https://github.com/edamontology/edam-bioimaging).

* `imaging_subontology.json`: Describes which subset of the full EDAM ontology will be recognized. See Python method `ontomatch.data.ontology.Ontology.get_subontology()`.

* `curated_imaging_synonyms-220316.csv`: Set of curated synonyms for imaging sub-ontology terms. Format is for reading by the Python function `ontomatch.data.imgontology.read_curated_synonyms()`.

