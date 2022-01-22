# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_dataset.ipynb (unless otherwise specified).

__all__ = ['URL', 'CMUDict']

# Cell

import os
import datasets
import torchtext
import csv

URL= 'https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict'

_DESCRIPTION = """\
Books are a rich source of both fine-grained information, how a character, \
an object or a scene looks like, as well as high-level semantics, what \
someone is thinking, feeling and how these states evolve through a story.\
This work aims to align books to their movie releases in order to provide\
rich descriptive explanations for visual content that go semantically far\
beyond the captions available in current datasets. \
"""


_CITATION = """\
@InProceedings{Zhu_2015_ICCV,
    title = {Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books},
    author = {Zhu, Yukun and Kiros, Ryan and Zemel, Rich and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015}
}
"""

# Cell
import re

_PUNCTUATIONS = set([
    "!EXCLAMATION-POINT",
    "\"CLOSE-QUOTE",
    "\"DOUBLE-QUOTE",
    "\"END-OF-QUOTE",
    "\"END-QUOTE",
    "\"IN-QUOTES",
    "\"QUOTE",
    "\"UNQUOTE",
    "#HASH-MARK",
    "#POUND-SIGN",
    "#SHARP-SIGN",
    "%PERCENT",
    "&AMPERSAND",
    "'END-INNER-QUOTE",
    "'END-QUOTE",
    "'INNER-QUOTE",
    "'QUOTE",
    "'SINGLE-QUOTE",
    "(BEGIN-PARENS",
    "(IN-PARENTHESES",
    "(LEFT-PAREN",
    "(OPEN-PARENTHESES",
    "(PAREN",
    "(PARENS",
    "(PARENTHESES",
    ")CLOSE-PAREN",
    ")CLOSE-PARENTHESES",
    ")END-PAREN",
    ")END-PARENS",
    ")END-PARENTHESES",
    ")END-THE-PAREN",
    ")PAREN",
    ")PARENS",
    ")RIGHT-PAREN",
    ")UN-PARENTHESES",
    "+PLUS",
    ",COMMA",
    "--DASH",
    "-DASH",
    "-HYPHEN",
    "...ELLIPSIS",
    ".DECIMAL",
    ".DOT",
    ".FULL-STOP",
    ".PERIOD",
    ".POINT",
    "/SLASH",
    ":COLON",
    ";SEMI-COLON",
    ";SEMI-COLON(1)",
    "?QUESTION-MARK",
    "{BRACE",
    "{LEFT-BRACE",
    "{OPEN-BRACE",
    "}CLOSE-BRACE",
    "}RIGHT-BRACE",
])

_alt_re = re.compile(r'[^a-zA-Z]+')

class CMUDict(datasets.GeneratorBasedBuilder):
    """ CMU Dict dataset """

    BUILDER_CONFIGS=[
        datasets.BuilderConfig(name='cmu3',description='cmu phonemes to words', version='1.0.0')
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features = datasets.Features(
                {
                    'word': datasets.Value("string"),
                    'word_length': datasets.Value('uint8'),
                    'phoneme': datasets.Sequence(datasets.Value("string"))
                }
            ),
            supervised_keys=None,
            citation=_CITATION
        )

    def _vocab_text_gen(self, archive):
        for _,ex in self._generate_examples(archive):
            yield ex['text']

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': data_dir, 'split': 'train'}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'filepath': data_dir, 'split': 'validation'}
            )
        ]

    def _generate_examples(self, filepath, split='train'):
        with open(filepath, encoding='utf-8') as csv_file:
            with open(filepath, encoding='utf-8') as f:
                for _id, line in enumerate(f):
                    if not line or line.startswith(';;;'):  # ignore comments
                        continue
                    word, *phones = line.strip().split(' ')
                    if word in _PUNCTUATIONS:
                        if exclude_punctuations:
                            continue

                        if word.startswith("..."):
                            word = '...'

                        if word.startswith("--"):
                            word = '--'
                        else:
                            word = word[0]

                    # if word has multiple pronounciations, there will be (number appended to it)
                    # for example, DATAPOINTS DATAPOINTS(1)
                    # regular expression _alt_re removes (1) and change DATAPOINTS(1) to DATAPOINTS
                    word = re.sub(_alt_re, '', word)
                    yield _id, {
                        'word': word,
                        'word_length': len(word),
                        'phoneme': phones
                    }