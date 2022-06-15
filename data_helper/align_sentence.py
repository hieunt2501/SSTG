import re
from edlib import align, getNiceAlignment
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def smoke_cigar(cigar, src=[], trg=[]):
    ans = []
    number = 0
    query_align = 0
    target_align = 0
    for c in cigar:
        if c.isdigit():
            number = number * 10
            number += int(c)
        else:
            if c == "=":
                ans.extend([(query_align + i, target_align + i)
                           for i in range(number)])
                query_align += number
                target_align += number
            elif c == "I":
                ans.append((-1, target_align))
                target_align += 1
            elif c == "D":
                ans.append((query_align, -1))
                query_align += 1
            else:
                ans.append((query_align, target_align))
                target_align += 1
                query_align += 1
            number = 0
    if src:
        assert len(ans) == len(src)

    return ans


TOKENIZER_REGEX = re.compile(r'(\W)')


def tokenizer(text):
    tokens = TOKENIZER_REGEX.split(text)
    return [t for t in tokens if len(t.strip()) > 0]


def getNiceAlignment(alignResult, query, target, gapSymbol="-"):
    target_pos = alignResult["locations"][0][0]
    if target_pos == None:
        target_pos = 0
    query_pos = 0  # 0-indexed
    target_aln = match_aln = query_aln = []
    cigar = alignResult["cigar"]
    tags = []
    toks = []
    trgs = []
    index = 0
    for num_occurrences, alignment_operation in re.findall("(\d+)(\D)", cigar):
        num_occurrences = int(num_occurrences)
        # print(num_occurrences, alignment_operation)
        if alignment_operation == "=":
            tar = target[target_pos: target_pos + num_occurrences]
            target_pos += num_occurrences
            que = query[query_pos: query_pos + num_occurrences]
            query_pos += num_occurrences
            for t, q in zip(tar, que):
                tags.append("KEEP")
                toks.append(t)
                trgs.append('')
                index += 1
            # match_aln += "|" * num_occurrences
        elif alignment_operation == "X":
            tar = target[target_pos: target_pos + num_occurrences]
            target_pos += num_occurrences
            que = query[query_pos: query_pos + num_occurrences]
            query_pos += num_occurrences
            for t, q in zip(tar, que):
                tags.append(f"CHANGE")
                toks.append(t)
                trgs.append(q)
                index += 1
            # match_aln += "." * num_occurrences
        elif alignment_operation == "D":
            tar = target[target_pos: target_pos + num_occurrences]
            target_pos += num_occurrences
            for t in tar:
                tags.append("DELETE")
                toks.append(t)
                trgs.append('')
                index += 1
        elif alignment_operation == "I":
            que = query[query_pos: query_pos + num_occurrences]
            query_pos += num_occurrences
            # if trgs[-1] == '':
            # trgs[-1] += toks[-1]
            for q in que:
                trgs[-1] += ' ' + q
            tags[-1] = "APPEND"
        else:
            raise Exception(
                "The CIGAR string from alignResult contains a symbol not '=', 'X', 'D', 'I'. Please check the validity of alignResult and alignResult.cigar")

    return tags, toks, trgs


def align_sentence(sent1, sent2):
    """from sent1 -> sent2"""
    # s1 = sent1.split()
    # s2 = sent2.split()
    s1 = text_to_word_sequence(sent1)
    s2 = text_to_word_sequence(sent2)

    return getNiceAlignment(align(s2, s1, task="path"), s2, s1)
