
import ander.env
from .prepare_data import main

PARALLEL_SRC_TGT = (
    ("europarlv8/europarl-v8.fi-en.en", "europarlv8/europarl-v8.fi-en.fi"),
    ("wiki-titles/fi-en/titles.fi-en.en", "wiki-titles/fi-en/titles.fi-en.fi"),
)

PARALLEL_AUX_TGT = (
    ("europarlv7/europarl-v7.fr-en.en", "europarlv7/europarl-v7.fr-en.fr"),
    ("commoncrawl/commoncrawl.fr-en.en", "commoncrawl/commoncrawl.fr-en.fr"),
    ("undoc_2000/undoc.2000.fr-en.en", "undoc_2000/undoc.2000.fr-en.fr"),
)

MONOLINGUAL_SRC = (
    'news.2014/fi.shuffled',
)

VALIDATION_SET = (
    ('wmt15/dev/newsdev2015-enfi-src.en.sgm', 'wmt15/dev/newsdev2015-enfi-ref.fi.sgm'),
)

TEST_SET = (
    ('wmt15/test/newstest2015-enfi-src.en.sgm', 'wmt15/test/newstest2015-enfi-ref.fi.sgm'),
)

L_S, L_A, L_T = "en", "fr", "fi"


if __name__ == '__main__':
    name = 'nmt_aux_wmt15_fi'
    cwd = ander.env.set_workspace(name=name, persistent=True, scriptname=False)
    print(cwd)
    main(L_S, L_T, L_A, VALIDATION_SET, TEST_SET, PARALLEL_SRC_TGT, PARALLEL_AUX_TGT, MONOLINGUAL_SRC, reverse=True)
