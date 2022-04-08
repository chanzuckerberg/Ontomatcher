from setuptools import setup


setup(
    name='Ontomatcher',
    version='0.9',
    description='A general purpose, configurable, fast mention detector of ontology terms (entities) in plain text.',
    packages=['ontomatch', 'ontomatch.data', 'ontomatch.text', 'ontomatch.utils', 'ontomatch.nprhub'],
    url='https://github.com/chanzuckerberg/Ontomatcher',
    license='MIT',
    author='Sunil Mohan, CZI Science',
    author_email='smohan@chanzuckerberg.com',
    python_requires='>=3.8',
    install_requires=[
        'nltk >= 3.5',
        'numpy',
        'pygtrie >= 2.4.2',
        'unidecode >= 1.3.4'
    ]
)
