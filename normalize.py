"""
Normalize a video script (.md/.txt) into TTS-ready narration text.

    uv run python normalize.py ../jk-video/xxx/video_script_final.md
    uv run python normalize.py script.md -o input/xxx.txt
    uv run python normalize.py script.md --lang zh      # override detection
    uv run python normalize.py script.md --no-llm       # rules only, offline
    uv run python normalize.py script.md --llm ollama   # pin one backend
    uv run python normalize.py --self-check

Does three things:
  1. keeps only the spoken blocks (drops B-roll / on-screen data / tables / metadata)
  2. respells what TTS mispronounces, in the language the script is written in:
       en  AES67   -> "A-E-S six seven"      4:4:4 -> "four four four"
       zh  AES67   -> "A E S 六七"            18%   -> "百分之十八"
     An English-only script never gets Chinese readings, and vice versa.
  3. asks an LLM how to say whatever the rules could not place (claude -> codex
     -> local ollama, first one that answers), remembers it in
     lexicon_learned.json, and prints everything it changed

The source script stays clean; this writes a separate <stem>_tts.txt.
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Readings the LLM worked out on earlier runs, so each term is asked about once.
# Hand-written LEXICON entries below always win over anything learned here.
LEARNED_PATH = Path(__file__).parent / "lexicon_learned.json"

# Tried in order, first usable answer wins: subscription CLIs first, local last.
# --llm claude|codex|ollama pins one; a missing binary just falls through.
LLM_BACKENDS = {
    "claude": ["claude", "-p", "--model", "haiku"],
    "codex": ["codex", "exec", "--skip-git-repo-check", "--color", "never"],
    "ollama": ["ollama", "run", "qwen3.5:9b"],
}

LLM_PROMPT = """You prepare scripts for a text-to-speech engine that will read \
them aloud in {language}. For each term below, reply with the spelling that makes \
the engine pronounce it the way an industry professional says it out loud.

Rules:
- Spell out letter-by-letter as {letter_example} only when the letters are said \
individually. If it is said as a word (SaaS -> sass, JSON -> jason), write that word.
- Write numbers as words the way they are spoken: {number_example}
- Never translate, never explain, never add or drop information.
- If the term is an ordinary word or a brand said as written, repeat it unchanged.

Reply with one JSON object mapping each term to its reading. No other text.

Terms: {terms}"""

# Explicit readings, per language. Highest priority, checked before any regex.
# THIS IS THE TUNING KNOB -- when the audio sounds wrong, add the term here.
LEXICON = {
    "en": {
        "AES67": "A-E-S six seven",
        "eARC": "E-A-R-C",   # "E-ARC" came back from ASR as "ER key"
        "OAuth": "O-auth",
        "PoE": "P-O-E",
        "RTP": "R-T-P",
        "CAT5e": "cat five E",
        "CAT6": "cat six",
        "SaaS": "sass",
        "SQL": "sequel",
        "JSON": "jason",
        "GUI": "gooey",
        "Wi-Fi": "why-fye",
        "vs.": "versus",
        "vs": "versus",
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "et cetera",
        "AI": "AI",     # listed so it is not flagged for review
        "CEO": "CEO",
        "API": "API",
        "USB": "USB",
    },
    "zh": {
        "AES67": "A E S 六七",
        "eARC": "E A R C",
        "OAuth": "O Auth",
        "SaaS": "萨斯",
        "API": "A P I",
        "GPT": "G P T",
        "UI": "U I",
        "URL": "U R L",
        "CLI": "C L I",
        "SDK": "S D K",
        "CPU": "C P U",
        "GPU": "G P U",
        "LLM": "L L M",
        "MCP": "M C P",
        "HDMI": "H D M I",
        "IPO": "I P O",
        "ROI": "R O I",
        "vs": "对比",
        "AI": "AI",
    },
}

# How to write a spelled-out acronym so the engine actually says the letters.
# Measured, not guessed: synthesize each variant and transcribe it back.
#   Qwen3-TTS (jk_tts)  "A-E-S"   -> AES67 heard correctly
#   CosyVoice           "A-E-S"   -> "AE67", the S vanishes; "A.E.S." is clean
# A consumer overrides it with normalize.LETTER_SEP = "." before calling.
LETTER_SEP = "-"

# Acronyms a voice would slur into a word instead of spelling out.
SPELL_OUT = {"AES", "EDID", "ARC", "CEC", "POE", "SDR", "HDR", "RAG", "SAM", "ARM"}

# zh only: an English TTS reads these correctly on its own.
ZH_UNITS = {"Hz": "赫兹", "kHz": "千赫", "MHz": "兆赫", "GHz": "吉赫",
            "W": "瓦", "kW": "千瓦", "V": "伏"}

# Tags that open a spoken block. Anything under another [tag], a heading,
# a rule or a table row is treated as production notes, not speech.
NARRATION_TAGS = ("[Narration]", "[VO]", "[Voiceover]", "[Script]", "[口播]")

EN_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
             "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
CN_DIGITS = "零一二三四五六七八九"

CJK = re.compile(r"[一-鿿]")


def detect_lang(text: str) -> str:
    """zh once Chinese is a real part of the script, not a stray quoted term."""
    dense = re.sub(r"\s", "", text)
    return "zh" if dense and len(CJK.findall(dense)) / len(dense) > 0.05 else "en"


def restyle(reading: str) -> str:
    """LEXICON is written in the canonical hyphen style ("A-E-S six seven");
    rewrite its letter runs into whatever separator this engine needs."""
    if LETTER_SEP == "-":
        return reading
    return re.sub(r"(?:[A-Z]-)+[A-Z]",
                  lambda m: spell(m.group(0).replace("-", ""), "en"), reading)


def spell(letters: str, lang: str) -> str:
    """AES -> A-E-S / A.E.S. / A E S, depending on what the engine understands."""
    sep = " " if lang == "zh" else LETTER_SEP
    out = sep.join(letters.upper())
    return out + sep if sep == "." else out


def say_digits(d: str, lang: str) -> str:
    """67 -> six seven / 六七 -- model numbers are read digit by digit"""
    if lang == "en":
        return " ".join(EN_DIGITS[c] for c in d)
    return "".join(CN_DIGITS[int(c)] for c in d)


def cn_number(n: int) -> str:
    """67 -> 六十七 (a value, not a model number). Digit-by-digit above 9999."""
    if n < 10:
        return CN_DIGITS[n]
    if n > 9999:
        return say_digits(str(n), "zh")
    units = ["", "十", "百", "千"]
    s = str(n)
    out: list[str] = []
    pending_zero = False
    for i, ch in enumerate(s):
        d = int(ch)
        if d == 0:
            pending_zero = True
            continue
        if pending_zero and out:
            out.append("零")
        pending_zero = False
        out.append(CN_DIGITS[d] + units[len(s) - i - 1])
    r = "".join(out)
    return r[1:] if r.startswith("一十") else r


def extract_narration(text: str) -> str:
    """Keep only the spoken blocks. A file with no narration tag is all speech."""
    if not any(tag in text for tag in NARRATION_TAGS):
        return text
    out, speaking = [], False
    for line in text.splitlines():
        s = line.strip()
        tag = next((t for t in NARRATION_TAGS if s.startswith(t)), None)
        if tag:
            speaking = True
            rest = s[len(tag):].strip()
            if rest:
                out.append(rest)
            continue
        if s.startswith(("[", "#", "---", "|", ">")):
            speaking = False
            continue
        if speaking:
            out.append(line)
    return "\n".join(out)


def strip_markdown(text: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links/images
    text = re.sub(r"[*_`~]+", "", text)
    text = re.sub(r"^\s{0,3}(#{1,6}|[-+*]|\d+\.)\s+", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def one_sentence_per_paragraph(text: str) -> str:
    """process.py only groups on blank lines and never splits on an English
    period, so a paragraph of English prose reaches the model as one oversized
    segment. Hand it sentences it can pack itself."""
    parts = re.split(r"(?<=[.!?。！？；])(?<![A-Z]\.)\s+(?=[A-Z\u4e00-\u9fff])",
                     text.replace("\n", " "))
    return "\n\n".join(p.strip() for p in parts if p.strip())


def load_learned() -> dict:
    if not LEARNED_PATH.exists():
        return {"en": {}, "zh": {}}
    data = json.loads(LEARNED_PATH.read_text(encoding="utf-8"))
    return {"en": data.get("en", {}), "zh": data.get("zh", {})}


def ask_llm(terms: list[str], lang: str, backend: str | None = None) -> dict:
    """Ask an LLM how each unfamiliar term is said aloud. Never fatal:
    every backend failing just leaves the rule output as-is."""
    prompt = LLM_PROMPT.format(
        language="Chinese" if lang == "zh" else "English",
        letter_example="A E S" if lang == "zh" else "A-E-S",
        number_example="67 -> 六七" if lang == "zh" else "67 -> six seven",
        terms=", ".join(terms),
    )
    names = [backend] if backend else list(LLM_BACKENDS)
    for name in names:
        cmd = LLM_BACKENDS[name]
        try:
            r = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=180)
            # ollama emits <think> blocks; every CLI wraps the JSON in chatter
            blob = re.search(r"\{[^{}]*\}", re.sub(r"<think>.*?</think>", "", r.stdout, flags=re.S), re.S)
            readings = _keep_readings(json.loads(blob.group(0)), terms, lang) if blob else {}
        except Exception as e:
            print(f"  ({name} unusable: {e})")
            continue
        if readings:
            print(f"  (via {name})")
            return readings
        print(f"  ({name} returned nothing usable)")
    return {}


def _keep_readings(raw: dict, terms: list[str], lang: str) -> dict:
    """An LLM reply is untrusted input: a reading may only respell one short
    term, never smuggle in prose, punctuation, or the wrong language."""
    ok = {}
    for term, reading in raw.items():
        reading = str(reading).strip()
        if term not in terms or not reading:
            continue
        if len(reading) > 4 * len(term) + 20:
            continue
        if not re.fullmatch(r"[\w\s.\-一-鿿]+", reading):
            continue
        if lang == "en" and CJK.search(reading):
            continue
        ok[term] = reading
    return ok


def normalize(text: str, lang: str | None = None,
              learned: dict | None = None) -> tuple[str, str, Counter, Counter]:
    """Returns (normalized text, lang used, substitutions, tokens left for review)."""
    lang = lang or detect_lang(text)
    lexicon = {**(learned or {}).get(lang, {}), **LEXICON[lang]}
    if lang == "en":
        lexicon = {k: restyle(v) for k, v in lexicon.items()}
    subs: Counter = Counter()

    def sub(pattern, repl, flags=0):
        nonlocal text

        def _r(m):
            new = repl(m)
            if new != m.group(0):
                subs[f"{m.group(0)} -> {new}"] += 1
            return new

        text = re.sub(pattern, _r, text, flags=flags)

    # 1. lexicon, longest term first so AES67 wins over AES
    for term in sorted(lexicon, key=len, reverse=True):
        sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            lambda m, t=term: lexicon[t])

    # 2. a:b:c ratios -- 4:4:4 would otherwise be read as a timestamp
    sub(r"(?<![\d:])(\d{1,2}(?::\d{1,2}){2,3})(?![\d:])",
        lambda m: ("比" if lang == "zh" else " ").join(
            say_digits(p, lang) for p in m.group(1).split(":")))

    # 3. model codes -- JTD-2860, HDMI2.1, RTX4090, 8K60
    sub(r"(?<![A-Za-z0-9-])([A-Za-z]{2,6})[-–]?(\d{1,6})(\.\d)?(?![A-Za-z0-9-])",
        lambda m: f"{spell(m.group(1), lang)} {say_digits(m.group(2), lang)}"
                  + ((("点" + CN_DIGITS[int(m.group(3)[1])]) if lang == "zh"
                      else " point " + EN_DIGITS[m.group(3)[1]]) if m.group(3) else ""))

    # 4. acronyms a voice would slur into a word
    sub(rf"(?<![A-Za-z0-9-])({'|'.join(sorted(SPELL_OUT, key=len, reverse=True))})(?![A-Za-z0-9-])",
        lambda m: spell(m.group(1), lang))

    if lang == "zh":
        # 5. an English voice already reads these right; a Chinese one does not
        sub(r"(\d+(?:\.\d+)?)\s?%",
            lambda m: "百分之" + (cn_number(int(float(m.group(1))))
                                 if float(m.group(1)).is_integer()
                                 else m.group(1).replace(".", "点")))
        sub(rf"(\d+)\s?({'|'.join(sorted(ZH_UNITS, key=len, reverse=True))})(?![A-Za-z])",
            lambda m: cn_number(int(m.group(1))) + ZH_UNITS[m.group(2)])

    text = re.sub(r"(?<=[A-Z]\.)\.", "", text)   # "P.O.E.." -> "P.O.E."

    # whatever is still SHOUTY or alphanumeric is a pronunciation risk -- report it,
    # minus the A-B-C forms the rules above already spelled out
    review = Counter(
        t for t in re.findall(r"(?<![A-Za-z0-9-])[A-Za-z0-9][A-Za-z0-9.\-]*", text)
        # "A-B-C" anywhere in the token means the rules already spelled it,
        # suffix and all ("P-O-E-enabled") -- asking about it teaches nonsense
        if not re.search(r"[A-Z][-.][A-Z]", t) and t not in lexicon
        # two capitals means an acronym even when lowercase letters hide in it
        # (PoE, SDVoE, eARC), and any letter+digit mix is a model number
        and (len(re.findall(r"[A-Z]", t)) > 1
             or (re.search(r"[A-Za-z]", t) and re.search(r"\d", t)))
    )
    return text, lang, subs, review


def run(src: Path, dst: Path | None, lang: str | None,
        use_llm: bool = True, backend: str | None = None) -> Path:
    raw = src.read_text(encoding="utf-8")
    body = strip_markdown(extract_narration(raw))
    learned = load_learned()
    out, lang, subs, review = normalize(body, lang, learned)

    # Anything the rules could not place goes to the LLM once, then is remembered.
    fresh = {}
    if use_llm and review:
        print(f"asking the LLM about {len(review)} term(s): {', '.join(review)}")
        fresh = ask_llm(list(review), lang, backend)
        if fresh:
            learned[lang].update(fresh)
            LEARNED_PATH.write_text(json.dumps(learned, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
            out, lang, subs, review = normalize(body, lang, learned)

    out = one_sentence_per_paragraph(out)

    dst = dst or src.with_name(f"{src.stem}_tts.txt")
    dst.write_text(out + "\n", encoding="utf-8")

    print(f"{src.name} -> {dst}  (lang={lang}, {len(raw)} -> {len(out)} chars)")
    if subs:
        print("\nrespelled:")
        for k, n in subs.most_common():
            print(f"  {n:3d}x  {k}")
    if fresh:
        print(f"\nlearned (saved to {LEARNED_PATH.name}, edit or delete to retune):")
        for k, v in fresh.items():
            print(f"       {k} -> {v}")
    if review:
        print("\nstill risky -- add to LEXICON if the voice gets it wrong:")
        for k, n in review.most_common(30):
            print(f"  {n:3d}x  {k}")
    return dst


def self_check() -> None:
    en = "The AES67 spec and JTD-2860 both do HDMI2.1"
    t, lang, _, _ = normalize(en)
    assert lang == "en"
    assert t == ("The A-E-S six seven spec and J-T-D two eight six zero "
                 "both do H-D-M-I two point one"), t
    assert not CJK.search(t), "English script must never get Chinese readings"

    t, _, _, _ = normalize("Chroma is 4:4:4 here, EDID passthrough over eARC")
    assert t == "Chroma is four four four here, E-D-I-D passthrough over E-A-R-C", t

    # English prose the engine already reads correctly stays untouched
    t, _, subs, _ = normalize("It went up 18% last quarter, roughly 60 watts.")
    assert t == "It went up 18% last quarter, roughly 60 watts." and not subs, t

    zh = "这台 JTD-2860 支持 HDMI2.1，色彩 4:4:4，带宽涨了 18%，功耗 60W"
    t, lang, _, _ = normalize(zh)
    assert lang == "zh"
    assert t == ("这台 J T D 二八六零 支持 H D M I 二点一，色彩 四比四比四，"
                 "带宽涨了 百分之十八，功耗 六十瓦"), t

    # one quoted English term does not make a Chinese script, and the reverse
    assert detect_lang("We tested the 中文 build once across a long English line.") == "en"
    assert detect_lang("我们今天聊 Claude Code 的 agent 能力") == "zh"

    assert cn_number(67) == "六十七" and cn_number(10) == "十"

    laid = one_sentence_per_paragraph("One. Two! Three? Done.")
    assert laid == "One.\n\nTwo!\n\nThree?\n\nDone.", laid
    assert one_sentence_per_paragraph("HDMI two point one is fine.") == \
        "HDMI two point one is fine."

    _, _, _, done = normalize("Use a PoE-enabled switch and an eARC-capable display.")
    assert not done, f"already-spelled tokens must not be re-asked: {done}"

    _, _, _, flagged = normalize("An SDVoE box, a PoE switch, one 10GbE trunk, plain words")
    assert set(flagged) >= {"SDVoE", "10GbE"}, flagged   # mixed-case acronyms count
    assert "plain" not in flagged and "box" not in flagged

    global LETTER_SEP
    LETTER_SEP = "."
    t, _, _, dotted = normalize("Select the RTP tab, enable AES67 over PoE.")
    assert t == "Select the R.T.P. tab, enable A.E.S. six seven over P.O.E.", t
    assert not dotted, f"dotted spellings must not be re-asked: {dotted}"
    assert one_sentence_per_paragraph("Open the R.T.P. Config tab. Then click Add.") == \
        "Open the R.T.P. Config tab.\n\nThen click Add."
    LETTER_SEP = "-"

    good = {"NDI": "N-D-I", "Dante": "Dante"}
    bad = {"NDI": "N-D-I. This term refers to Network Device Interface, which is",
           "RTP": "R-T-P", "NDI": "N D I 一二三"}
    assert _keep_readings(good, ["NDI", "Dante"], "en") == good
    assert _keep_readings(bad, ["NDI"], "en") == {}, "prose and CJK must be rejected"

    md = "## Title\n\n[VO]\n**Hey folks**, let's talk API.\n\n[B-roll] server room\n\n| a | b |\n"
    assert strip_markdown(extract_narration(md)) == "Hey folks, let's talk API."

    print("self-check ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--self-check" in args:
        self_check()
        sys.exit(0)
    if not args:
        print(__doc__)
        sys.exit(1)
    out = lang = None
    backend = None
    for flag in ("-o", "--lang", "--llm"):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1]
            if flag == "-o":
                out = Path(value)
            elif flag == "--lang":
                lang = value
            else:
                backend = value
            del args[i:i + 2]
    if lang not in (None, "en", "zh"):
        print("--lang must be en or zh")
        sys.exit(1)
    if backend not in (None, *LLM_BACKENDS):
        print(f"--llm must be one of {', '.join(LLM_BACKENDS)}")
        sys.exit(1)
    use_llm = "--no-llm" not in args
    args = [a for a in args if a != "--no-llm"]
    run(Path(args[0]), out, lang, use_llm, backend)
