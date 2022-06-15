import re
import Levenshtein
import numpy as np

from difflib import SequenceMatcher
from tensorflow.keras.preprocessing.text import text_to_word_sequence

TOKENIZER_REGEX = re.compile(r'(\W)')

NER_KEEP = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}


def is_sent_ok(sent, delimeters=SEQ_DELIMETERS):
    for del_val in delimeters.values():
        if del_val in sent and del_val != delimeters["tokens"]:
            return False
    return True


def check_split(source_token, target_tokens):
    if source_token.split("-") == target_tokens:
        return "$TRANSFORM_SPLIT_HYPHEN"
    else:
        return None
    
def apply_transformation(source_token, target_token):
    target_tokens = text_to_word_sequence(target_token)
    if len(target_tokens) > 1:
        transform = check_split(source_token, target_tokens)
        if transform:
            return transform
    return None


def perfect_align(t, T, insertions_allowed=0,
                  cost_function=Levenshtein.distance):
    # dp[i, j, k] is a minimal cost of matching first `i` tokens of `t` with
    # first `j` tokens of `T`, after making `k` insertions after last match of
    # token from `t`. In other words t[:i] aligned with T[:j].

    # Initialize with INFINITY (unknown)
    shape = (len(t) + 1, len(T) + 1, insertions_allowed + 1)
    dp = np.ones(shape, dtype=int) * int(1e9)
    come_from = np.ones(shape, dtype=int) * int(1e9)
    come_from_ins = np.ones(shape, dtype=int) * int(1e9)

    dp[0, 0, 0] = 0  # The only known starting point. Nothing matched to nothing.
    for i in range(len(t) + 1):  # Go inclusive
        for j in range(len(T) + 1):  # Go inclusive
            for q in range(insertions_allowed + 1):  # Go inclusive
                if i < len(t):
                    # Given matched sequence of t[:i] and T[:j], match token
                    # t[i] with following tokens T[j:k].
                    for k in range(j, len(T) + 1):
                        transform = \
                            apply_transformation(t[i], '   '.join(T[j:k]))
                        if transform:
                            cost = 0
                        else:
                            cost = cost_function(t[i], '   '.join(T[j:k]))
                        current = dp[i, j, q] + cost
                        if dp[i + 1, k, 0] > current:
                            dp[i + 1, k, 0] = current
                            come_from[i + 1, k, 0] = j
                            come_from_ins[i + 1, k, 0] = q
                if q < insertions_allowed:
                    # Given matched sequence of t[:i] and T[:j], create
                    # insertion with following tokens T[j:k].
                    for k in range(j, len(T) + 1):
                        cost = len('   '.join(T[j:k]))
                        current = dp[i, j, q] + cost
                        if dp[i, k, q + 1] > current:
                            dp[i, k, q + 1] = current
                            come_from[i, k, q + 1] = j
                            come_from_ins[i, k, q + 1] = q

    # Solution is in the dp[len(t), len(T), *]. Backtracking from there.
    alignment = []
    i = len(t)
    j = len(T)
    q = dp[i, j, :].argmin()
    while i > 0 or q > 0:
        is_insert = (come_from_ins[i, j, q] != q) and (q != 0)
        j, k, q = come_from[i, j, q], j, come_from_ins[i, j, q]
        if not is_insert:
            i -= 1

        if is_insert:
            alignment.append(['INSERT', T[j:k], (i, i)])
        else:
            alignment.append([f'CHANGE_{t[i]}', T[j:k], (i, i + 1)])

    assert j == 0

    return dp[len(t), len(T)].min(), list(reversed(alignment))


def _split(token):
    if not token:
        return []
    parts = text_to_word_sequence(token)
    return parts or [token]


def convert_alignments_into_edits(alignment, shift_idx):
    edits = []
    action, target_tokens, new_idx = alignment
    source_token = action.replace("CHANGE_", "")

    # check if delete
    if not target_tokens:
        edit = [(shift_idx, 1 + shift_idx), "DELETE"]
        return [edit]

    # check splits
    for i in range(1, len(target_tokens)):
        target_token = " ".join(target_tokens[:i + 1])
        transform = apply_transformation(source_token, target_token)
        if transform:
            edit = [(shift_idx, shift_idx + 1), transform]
            edits.append(edit)
            target_tokens = target_tokens[i + 1:]
            for target in target_tokens:
                edits.append([(shift_idx, shift_idx + 1), f"CHANGE_{target}"])
            return edits

    transform_costs = []
    transforms = []
    for target_token in target_tokens:
        transform = apply_transformation(source_token, target_token)
        if transform:
            cost = 0
            transforms.append(transform)
        else:
            cost = Levenshtein.distance(source_token, target_token)
            transforms.append(None)
        transform_costs.append(cost)
    min_cost_idx = transform_costs.index(min(transform_costs))
    # append to the previous word
    for i in range(0, min_cost_idx):
        target = target_tokens[i]
        edit = [(shift_idx - 1, shift_idx), f"CHANGE_{target}"]
        edits.append(edit)
    # replace/transform target word
    transform = transforms[min_cost_idx]
    target = transform if transform is not None \
        else f"CHANGE_{target_tokens[min_cost_idx]}"
    edit = [(shift_idx, 1 + shift_idx), target]
    edits.append(edit)
    # append to this word
    for i in range(min_cost_idx + 1, len(target_tokens)):
        target = target_tokens[i]
        edit = [(shift_idx, 1 + shift_idx), f"CHANGE_{target}"]
        edits.append(edit)
    return edits


def convert_edits_into_labels(source_tokens, all_edits):
    # make sure that edits are flat
    flat_edits = []
    for edit in all_edits:
        (start, end), edit_operations = edit
        if isinstance(edit_operations, list):
            for operation in edit_operations:
                new_edit = [(start, end), operation]
                flat_edits.append(new_edit)
        elif isinstance(edit_operations, str):
            flat_edits.append(edit)
        else:
            raise Exception("Unknown operation type")
    all_edits = flat_edits[:]
    labels = []
    total_labels = len(source_tokens) + 1
    if not all_edits:
        labels = [["KEEP"] for x in range(total_labels)]
    else:
        for i in range(total_labels):
            edit_operations = [x[1] for x in all_edits if x[0][0] == i - 1
                               and x[0][1] == i]
            if not edit_operations:
                labels.append(["KEEP"])
            else:
                labels.append(edit_operations)
    return labels


def insert_dummy(tokens, p='[unused]'):
    rlt = []
    # cnt = 1
    for token in tokens:
        rlt.append(p)
        rlt.append(token)
        # cnt += 1
    rlt.append(p)
    return rlt


def align_sequences(source_sent, target_sent):
    # check if sent is OK
    if not is_sent_ok(source_sent) or not is_sent_ok(target_sent):
        return None

    source_tokens = text_to_word_sequence(source_sent)
    target_tokens = text_to_word_sequence(target_sent)
    # source_tokens = phobert_tokenize(source_sent)
    # target_tokens = phobert_tokenize(target_sent)
    # print(source_tokens)
    matcher = SequenceMatcher(None, source_tokens, target_tokens)
    diffs = list(matcher.get_opcodes())
    all_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        source_part = _split(" ".join(source_tokens[i1:i2]))
        target_part = _split(" ".join(target_tokens[j1:j2]))
        if tag == 'equal':
            continue
        elif tag == 'delete':
            # delete all words separatly
            for j in range(i2 - i1):
                edit = [(i1 + j, i1 + j + 1), 'DELETE']
                all_edits.append(edit)
        elif tag == 'insert':
            # append to the previous word
            for target_token in target_part:
                edit = ((i1 - 1, i1), f"CHANGE_{target_token}")
                all_edits.append(edit)
        else:
            # normalize alignments if need (make them singleton)
            _, alignments = perfect_align(source_part, target_part,
                                          insertions_allowed=0)
            for alignment in alignments:
                new_shift = alignment[2][0]
                edits = convert_alignments_into_edits(alignment,
                                                      shift_idx=i1 + new_shift)
                all_edits.extend(edits)

    # get labels
    labels = convert_edits_into_labels(source_tokens, all_edits)
    if len(labels) == len(source_tokens):
        return labels
    elif len(labels) - 1 == len(source_tokens):
        return labels[1:]
    else:
        return None

# class Tokenizer:
#     def __init__(self, mode):
#         self.mode = mode


#     def tokenize(self, sent):
#         if self.mode == "split":
#             return sent.split()
#         elif self.mode == "tf":
#             return text_to_word_sequence(sent)
#         elif self.mode == "custom":
#             tokens = TOKENIZER_REGEX.split(sent)
#             return [t for t in tokens if len(t.strip()) > 0]
#         elif self.mode == "phobert":
#             return tokenizer.tokenize(sent)


def convert_tags(source, target):
    source_tokens = insert_dummy(text_to_word_sequence(source))
    source = " ".join(source_tokens)
    labels = align_sequences(source, target)

    for idx, label in enumerate(labels):
        if len(label) > 1 and label[0].startswith("CHANGE"):
            labels[idx] = "CHANGE|" + "<|>".join([w.split("_")[1] for w in label])
        elif label[0].startswith("CHANGE"):
            labels[idx] = "CHANGE|" + label[0].split("_")[1]
        else:
            labels[idx] = label[0]
    return labels, source_tokens

def make_labels(tags, src):
    final_tags, final_src = [], []

    for idx, (tag, token) in enumerate(zip(tags, src)):
        if tag == "DELETE" and token == "[unused]":
            continue
        elif tag.startswith("CHANGE") and token == "[unused]":
            edits = tag[len("CHANGE|"):].split('<|>')
            if idx == 0: continue
            if final_tags[-1].startswith("CHANGE") or final_tags[-1].startswith("APPEND"):
                final_tags[-1] += '<|>' + '<|>'.join(edits)
            elif final_tags[-1] == "DELETE":
                for j in range(len(final_tags)-2, -1, -1):
                    if final_tags[j] == "DELETE": continue
                    elif final_tags[j] == "KEEP":
                        final_tags[j] = "APPEND|" + "<|>".join(edits)
                        break
                    elif final_tags[j].startswith("CHANGE") or final_tags[j].startswith("APPEND"):
                        final_tags[j] += "<|>" + "<|>".join(edits) 
                        break
            elif final_tags[-1] == "KEEP":
                final_tags[-1] = f"APPEND|{'<|>'.join(edits)}"
        else:
            final_tags.append(tag)
            final_src.append(token)
    assert len(final_tags) == len(final_src)
    return final_tags, final_src


def process_ner(text, ner_tag):
    text_tokens = []
    tag_tokens = []
    # text = preprocess(text)
    tokens = text_to_word_sequence(text)

    i = 0 # token

    tmp_token = ""
    for idx in range(len(ner_tag)):
        tag = ner_tag[idx]
        
        tag_text = tag[0]
        # tag_text = preprocess(tag_text)

        ner = tag[-1]

        tmp_tokens = tag_text.split()
        # if len(tmp_tokens) == 1 and ner == "O" and tmp_tokens[0].isupper():
        for token in tmp_tokens:
            if token.isupper() and ner == "O":
                ner = "B-LOC"

        if ner in ["B-LOC", "I-LOC", "B-ORG", "I-ORG"]:
            count_not_upper = 0
            for token in tmp_tokens:
                if not re.match(r'\w*[A-Z]\w*', token):
                    count_not_upper += 1

            if count_not_upper == len(tmp_tokens):
                ner = "O"
        
        if ner not in NER_KEEP:
            ner = "O"
        tag_text_tokens = text_to_word_sequence(tag_text)
        # print(tag_text_tokens)
        
        # tmp_list = []
        
        for token in tag_text_tokens:
            if token == tokens[i]:
                text_tokens.append(token)
                tag_tokens.append(ner)
                i += 1
                tmp_token = ""
            elif token in tokens[i]:
                tmp_token += token
                if tmp_token == tokens[i]:
                    text_tokens.append(tmp_token)
                    tag_tokens.append(ner)
                    tmp_token = ""
                    i += 1
    
    return text_tokens, tag_tokens