
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path
from io import open
import hashlib
from subprocess import Popen, PIPE
from ander.formats import SGMParser
from tempfile import NamedTemporaryFile
import ander.env


PARALLEL_SRC_TGT = (
    ("europarlv7/europarl-v7.de-en.de", "europarlv7/europarl-v7.de-en.en"),
    ("commoncrawl/commoncrawl.de-en.de", "commoncrawl/commoncrawl.de-en.en"),
    #("newscommentary11/news-commentary-v11.de-en.de", "newscommentary11/news-commentary-v11.de-en.en"),
)

PARALLEL_AUX_TGT = (
    ("europarlv7/europarl-v7.fr-en.fr", "europarlv7/europarl-v7.fr-en.en"),
    ("commoncrawl/commoncrawl.fr-en.fr", "commoncrawl/commoncrawl.fr-en.en"),
    ("undoc_2000/undoc.2000.fr-en.fr", "undoc_2000/undoc.2000.fr-en.en"),
)

MONOLINGUAL_SRC = (
    'news.2014/de.shuffled',
)

VALIDATION_SET = (
    ('wmt15/dev/newstest2014-deen-src.de.sgm', 'wmt15/dev/newstest2014-deen-ref.en.sgm'),
)

TEST_SET = (
    ('wmt15/test/newstest2015-deen-src.de.sgm', 'wmt15/test/newstest2015-deen-ref.en.sgm'),
)

L_S, L_A, L_T = "de", "fr", "en"


def download_data():
    pass

TOKENIZER_SCRIPT_PATH = '/opt/moses/moses-scripts/tokenizer/tokenizer.perl'
if not os.path.isfile(TOKENIZER_SCRIPT_PATH):
    TOKENIZER_SCRIPT_PATH = '/cl/nltools/mosesdecoder/scripts/tokenizer/tokenizer.perl'
if not os.path.isfile(TOKENIZER_SCRIPT_PATH):
    TOKENIZER_SCRIPT_PATH = os.path.expanduser('~/tmp/dl4mt-material/data/tokenizer.perl')


def moses_tokenizer(lang, fout):
    p = Popen(('perl', TOKENIZER_SCRIPT_PATH, '-l', lang, '-threads', '8'), stdin=PIPE, stdout=fout)
    return p


def parallel_data(src, tgt, l1, l2):
    data_path = os.path.expanduser("~/Data")
    src = os.path.join(data_path, src)
    tgt = os.path.join(data_path, tgt)

    ftmp1 = NamedTemporaryFile(delete=False)
    ftmp2 = NamedTemporaryFile(delete=False)
    l1_tokenizer = moses_tokenizer(l1, ftmp1)
    l2_tokenizer = moses_tokenizer(l2, ftmp2)
    ftmp1.close()
    ftmp2.close()

    print("** tokenizing parallel data...")
    with open(src, 'r', encoding='utf-8') as sfd, open(tgt, 'r', encoding='utf-8') as tfd:
        slines = SGMParser.sentence_iterator(sfd) if src.endswith('sgm') else iter(sfd)
        tlines = SGMParser.sentence_iterator(tfd) if tgt.endswith('sgm') else iter(tfd)
        for line in slines:
            l1_tokenizer.stdin.write((line + '\n').encode('utf-8'))
            l2_tokenizer.stdin.write((next(tlines) + '\n').encode('utf-8'))

    l1_tokenizer.stdin.close()
    l2_tokenizer.stdin.close()
    l1_tokenizer.wait()
    l2_tokenizer.wait()

    print("** ready tokenized parallel data...")
    with open(ftmp1.name, 'r', encoding='utf-8') as sfd, open(ftmp2.name, 'r', encoding='utf-8') as tfd:
        tfd_it = iter(tfd)
        for line in sfd:
            yield line, next(tfd_it)
    os.unlink(ftmp1.name)
    os.unlink(ftmp2.name)


def mono_data(src, l_s=L_S):
    data_path = os.path.expanduser("~/Data")
    src = os.path.join(data_path, src)

    ftmp1 = NamedTemporaryFile(delete=False)
    l1_tokenizer = moses_tokenizer(l_s, ftmp1)
    ftmp1.close()

    print("** tokenizing parallel data...")
    with open(src, 'r', encoding='utf-8') as sfd:
        slines = SGMParser.sentence_iterator(sfd) if src.endswith('sgm') else iter(sfd)
        for line in slines:
            l1_tokenizer.stdin.write((line + '\n').encode('utf-8'))

    l1_tokenizer.stdin.close()
    l1_tokenizer.wait()

    print("** ready tokenized parallel data...")
    with open(ftmp1.name, 'r', encoding='utf-8') as sfd:
        for line in sfd:
            yield line, line
    os.unlink(ftmp1.name)


def tokenize(line):
    return line.split()


def max50(s):
    return 0 < len(s) <= 50


def max50unique(s, used=set()):
    if not max50(s):
        return False

    js = " ".join(s)
    csum = hashlib.md5(js.encode("utf-8")).hexdigest()

    if csum in used:
        return False

    used.add(csum)
    return True


def _process(fn_out, corpora, limit, separator='\t'):
    count = 0
    when1 = max50
    when2 = max50unique
    print("## outputting file '{}'...".format(fn_out))
    with open(fn_out, 'w', encoding='utf-8') as fd_out:
        for corpus in corpora:
            for l1, l2 in corpus:
                t1 = tokenize(l1)
                t2 = tokenize(l2)
                if limit and count >= limit:
                    break
                if when1(t1) and when2(t2):
                    count += 1
                    fd_out.write(" ".join(t1) + separator + " ".join(t2) + "\n")
    print("## done outputting file")


def main(l_s=L_S, l_t=L_T, l_a=L_A, validation_set=VALIDATION_SET, test_set=TEST_SET, train_set=PARALLEL_SRC_TGT,
         aux_set=PARALLEL_AUX_TGT, mono_set=MONOLINGUAL_SRC, reverse=False):
    print('preparing data...')
    if os.path.exists('DATA_PREPARATION_DONE'):
        return
    download_data()

    _process('valid.set', (parallel_data(pair[0], pair[1], l_s, l_t) for pair in validation_set), None)
    _process('test.set', (parallel_data(pair[0], pair[1], l_s, l_t) for pair in test_set), None)
    _process('train.set', (parallel_data(pair[0], pair[1], l_s, l_t) for pair in train_set), None)
    if reverse:
        _process('train.aux.set', (parallel_data(pair[0], pair[1], l_s, l_a) for pair in aux_set), None)
        _process('train.mono.set', (mono_data(mono, l_t) for mono in mono_set), None)
    else:
        _process('train.aux.set', (parallel_data(pair[0], pair[1], l_a, l_t) for pair in aux_set), None)
        _process('train.mono.set', (mono_data(mono, l_s) for mono in mono_set), None)

    with open('DATA_PREPARATION_DONE', 'w'):
        pass


if __name__ == '__main__':
    name = 'nmt_aux_wmt15'
    cwd = ander.env.set_workspace(name=name, persistent=True, scriptname=False)
    print(cwd)
    main(L_S, L_T, L_A, VALIDATION_SET, TEST_SET, PARALLEL_SRC_TGT, PARALLEL_AUX_TGT, MONOLINGUAL_SRC)
