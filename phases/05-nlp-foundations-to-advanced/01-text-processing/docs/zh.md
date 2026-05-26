# Text Processing — Tokenization, Stemming, Lemmatization

> Language is continuous. Models are discrete. Preprocessing is the bridge.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## The Problem

A model cannot read "The cats were running." It reads integers.

Every NLP system opens with the same three questions. Where does a word start. What is the root of the word. How do we treat "run", "running", "ran" as the same thing when it helps, and as different things when it doesn't.

Get tokenization wrong and the model learns from garbage. If your tokenizer treats `don't` as one token but `do n't` as two, the training distribution splits. If your stemmer collapses `organization` and `organ` to the same stem, topic modeling dies. If your lemmatizer needs part-of-speech context but you don't pass it, verbs get treated as nouns.

This lesson builds the three preprocessing primitives from scratch, then shows how NLTK and spaCy do the same work so you can see the tradeoffs.

## The Concept

Three operations. Each has a job and a failure mode.

**Tokenization** splits a string into tokens. "Token" is deliberately vague because the right granularity depends on the task. Word-level for classical NLP. Subword for transformers. Character for languages without whitespace.

**Stemming** chops suffixes with rules. Fast, aggressive, dumb. `running -> run`. `organization -> organ`. That second one is the failure mode.

**Lemmatization** reduces a word to its dictionary form using grammar knowledge. Slower, accurate, needs a lookup table or morphological analyzer. `ran -> run` (needs to know "ran" is past tense of "run"). `better -> good` (needs to know comparative forms).

Rule of thumb. Stem when speed matters and you can tolerate noise (search indexing, rough classification). Lemmatize when meaning matters (question answering, semantic search, anything the user will read).

## Build It

### Step 1: a regex word tokenizer

The simplest useful tokenizer splits on non-alphanumeric characters while keeping punctuation as its own tokens. Not perfect, not final, but it runs in one line.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Three patterns in order of precedence. Words with optional inner apostrophe (`don't`, `it's`). Pure numbers. Any single non-whitespace non-alphanumeric character as a standalone token (punctuation).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Failure modes to notice. `3pm` splits to `['3', 'pm']` because we alternated between letter runs and digit runs. Good enough for most tasks. URLs, emails, hashtags all break. For production, add patterns before the general ones.

### Step 2: a Porter stemmer (step 1a only)

The full Porter algorithm has five phases of rules. Step 1a alone covers the most frequent English suffixes and teaches the pattern.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Read the rules top-down. The `ies -> i` rule is why `ponies -> poni`, not `pony`. Real Porter has step 1b that would fix it. Rules compete. Earlier rules win. The order matters more than any single rule.

### Step 3: a lookup-based lemmatizer

Lemmatization proper needs morphology. A tractable teaching version uses a small lemma table and a fallback.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

The last case is the key teaching moment. `watched` is not in our table and our fallback only handles `ing`. Real lemmatization covers `ed`, irregular verbs, comparative adjectives, plurals with sound changes (`children -> child`). This is why production systems use WordNet, spaCy's morphologizer, or a full morphological analyzer.

### Step 4: pipe them together

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

The missing piece is a POS tagger. Phase 5 · 07 (POS Tagging) builds one. For now, default everything to `NOUN` and acknowledge the limitation.

## Use It

NLTK and spaCy ship the production versions. A few lines each.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` handles contractions, Unicode, edge cases your regex misses. `PorterStemmer` runs all five phases. `WordNetLemmatizer` needs the POS tag translated from NLTK's Penn Treebank scheme to WordNet's abbreviation set. The translation wiring above is the bit most tutorials skip.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy hides the whole pipeline behind `nlp(text)`. Tokenization, POS tagging, and lemmatization all run. Faster than NLTK at scale. More accurate out of the box. The tradeoff is that you cannot easily swap individual components.

### When to pick which

| Situation | Pick |
|-----------|------|
| Teaching, research, swapping components | NLTK |
| Production, multi-language, speed matters | spaCy |
| Transformer pipeline (you'll tokenize with the model's tokenizer anyway) | Use `tokenizers` / `transformers` and skip classical preprocessing |

### The two failure modes nobody warns you about

Most tutorials teach the algorithms and stop. Two things will bite a real preprocessing pipeline, and they are almost never covered.

**Reproducibility drift.** NLTK and spaCy change tokenization and lemmatizer behavior between versions. What produced `['do', "n't"]` in spaCy 2.x may produce `["don't"]` in 3.x. Your model was trained on one distribution. Inference now runs on a different one. Accuracy quietly degrades and nobody knows why. Pin library versions in `requirements.txt`. Write a preprocessing regression test that freezes expected tokenization of 20 sample sentences. Run it on every upgrade.

**Training / inference mismatch.** Train with aggressive preprocessing (lowercase, stopword removal, stemming), deploy on raw user input, watch performance crater. This is the single most common production NLP failure. If you preprocess during training, you must run the identical function during inference. Ship preprocessing as a function inside the model package, not as a notebook cell the serving team rewrites.

## Ship It

A reusable prompt that helps engineers pick a preprocessing strategy without reading three textbooks.

Save as `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize` 以将 URL 保留为单个令牌。测试：`tokenize("今天访问 https://example.com。")` 应该生成一个 URL 令牌。
2. **中。** 实施波特步骤 1b。如果单词包含元音并以“ed”或“ing”结尾，请将其删除。处理双辅音规则（“hopping -> hop”，而不是“hopp”）。
3. **难。** 构建一个词形还原器，它使用 WordNet 作为查找表，但当 WordNet 没有条目时回退到 Porter 词干分析器。对照普通 WordNet 和普通 Porter 测量标记语料库的准确性。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|-----------------|------------------------|
|代币|一句话|无论模型消耗多少单位。可以是字、子字、字符或字节。 |
|茎|词根|基于规则的后缀剥离的结果。并不总是一个真实的词。 |
|引理|词典形式 |您要查找的表格。需要语法上下文才能正确计算。 |
| POS 标签 |词类|名词、动词、形容词等类别。需要准确地进行词形还原。 |
|形态学|字形规则|单词如何根据时态、数字、大小写改变形式。词形还原取决于它。 |

## 进一步阅读

- [波特，M.F. (1980)。后缀剥离的算法](https://tartarus.org/martin/PorterStemmer/def.txt) - 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — 语言特征](https://spacy.io/usage/linguistic-features) — 真实管道的连接方式。
- [NLTK 书，第 3 章](https://www.nltk.org/book/ch03.html) — 您尚未想到的标记化边缘情况。
