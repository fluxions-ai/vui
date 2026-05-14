# Prompting Vui Nano

Two prompts go into every Vui generation: a **voice prompt** (audio + matching transcript that defines the speaker) and the **text** to render in that voice. Getting both right is more important than any sampling knob.

This doc is for people using Vui Nano standalone — `demo.py`, `engine.render()`, or driving `/v1/realtime` with their own LLM. If you're using the streaming server's bundled assistant, the [soul](soul.md) already encodes most of these rules; you only need this doc when writing your own.

## Voice prompts (`prompts/*.wav` + `*.txt`)

A voice prompt is a `.wav` of someone speaking + a `.txt` with the exact transcript of that wav. The model conditions on both — codec tokens from the audio, text tokens from the transcript — and then continues from there with whatever new text you hand it.

Three rules, in order of how much they matter:

1. **The text MUST match the audio.** Word for word, including filler ("um", "yeah"), hesitations, and any tags ([breath], [laugh]) that are audible in the clip. A mismatched transcript is the single most common reason cloned voices sound off — the model learns the wrong alignment and drifts. Run the audio through ASR if you're not sure; correct it by hand for filler and tags.

2. **Aim for 30 seconds or more.** The model has roughly **6 minutes** of context (4500 frames at 12.5 Hz), so the longer your prompt the more speaker character it locks onto. Below ~10 seconds, output quality and consistency drop sharply. 30s–2min is the sweet spot for everyday cloning; up to a minute or two helps a lot for distinctive accents.

3. **Garbage in, garbage out.** Whatever's in the prompt audio bleeds into the output — background noise, room reverb, compression artifacts, a poor mic, music in the background, low bitrate. Record in a quiet room with a decent mic at 24 kHz if you can. You can partly compensate at inference with the **SQ** conditioning channels (push toward 5 = clean) but it's not magic; clean source audio beats post-hoc cleanup every time.

The four preset voices shipped in `prompts/` (`maeve`, `abraham`, `rhian`, `harry`) follow all three rules and the released checkpoint is fine-tuned on them — they're what the model can do at its best. Arbitrary clones will trend toward those four's prosody.

## The text you ask it to speak

Vui Nano was trained on conversational speech with breaths, laughter, hesitations, and natural punctuation — not on read-aloud audiobook prose. Write text the way someone would actually speak it.

### Tags (inline, in square brackets)

Recognised tags are inserted inline and the model produces the matching audio:

| Tag | What it does |
|---|---|
| `[breath]` | Audible in-breath. Use sparingly between clauses or before a long sentence. |
| `[laugh]` | Short laugh. Works best mid-sentence (`"oh [laugh] yeah, totally"`). |
| `[sigh]` | Audible sigh. Good for resignation/fatigue. |
| `[gasp]` | Sharp intake. Surprise/shock. |
| `[cough]` | Self-explanatory; rarely needed. |
| `[hesitate]` | Brief filled pause. Pair with `um/uh` for naturalness. |

Tags **never** touch punctuation. Write `[gasp] Wait` — not `[gasp]! Wait` or `[gasp]Wait`. Multi-tag in one line is fine when the moment is emotionally heavy: `"[hesitate] yeah, [hesitate] I get that."`

Things the model was **not** trained on and that don't work: `haha`, `[breath]` written as `*breath*`, dropping `-ing` endings (`runnin'` → write `running` and let the speaker prompt handle the accent).

### Punctuation

Punctuation drives prosody. Read your text aloud — if you'd pause, the punctuation needs to mark it.

- **Periods** end complete sentences only. Don't drop a period mid-thought as a pause marker. `"It's heavy. salty. fatty stuff."` reads wrong — write `"It's heavy, salty, fatty stuff."`
- **Commas** for short beats: `"oh, mate, that's mad"`.
- **Ellipsis (`...`)** for hesitation, trailing-off, an unfinished thought: `"it's um... it's complicated"`, `"I just... yeah."`
- **Em-dash (`—`)** for self-interruption, sudden direction change: `"I tried that — but it didn't work"`, `"are you- are you okay?"`.
- **Hyphens** for word cutoffs and restarts: `"the- the thing"`, `"I th- I think"`, `"it's it's fine"`.

If a period would land before a lowercase word, it's wrong — replace with ellipsis, em-dash, or comma. The model was trained to break on terminal `.!?` so a bare period mid-clause makes it close prosody early and the TTS sounds clipped.

### Numbers, times, units, money

Spell **everything** phonetically. Vui Nano does not silently expand digits or symbols — it'll read `$50` as "dollar five zero" and `9:00 AM` as "nine colon zero zero AM".

| Bad | Good |
|---|---|
| `9:00 AM` | `nine in the morning` |
| `11:30` | `half eleven` |
| `$50` | `fifty quid` (or `fifty dollars`) |
| `$106.88` | `a hundred and six dollars eighty eight cents` |
| `Sept 3rd 2024` | `September third twenty twenty four` |
| `800MW` | `eight hundred megawatts` |
| `3.5%` | `three point five percent` |
| `60Hz` | `sixty hertz` |
| `10kg` | `ten kilograms` |
| `100km/h` | `a hundred kilometres an hour` |
| `GPT-4` | `GPT four` |
| `iPhone 15` | `iPhone fifteen` |
| `RTX 4090` | `RTX forty ninety` |
| `1993` | `nineteen ninety three` |

Times: drop `:00` on the hour (`eleven AM`, not `eleven oh oh`); use natural phrasing for half/quarter (`half eleven`, `quarter past two`); use `noon` / `midnight`. Years and 4-digit model numbers: read in pairs (`2024` → `twenty twenty four`).

If unsure how a unit is pronounced, spell out the full word (`megawatts`, not `MW`).

### Markdown, bullets, code

Don't. Vui Nano won't render `**bold**`, `# headings`, backticks, list bullets, or URLs — it'll either skip them or read them literally. Strip everything to prose before generation.

### Special characters

Be careful with anything that isn't plain ASCII. Curly quotes (`“ ” ‘ ’`), non-breaking spaces, zero-width joiners, unusual unicode dashes, accented characters the model rarely saw in training, and emoji are all silent landmines — they either trigger weird prosody, get skipped, or read literally ("smiling face emoji"). Normalise to plain ASCII where you can: straight quotes (`" "`), regular hyphen/em-dash (`-` / `—`), regular spaces. If you're piping LLM output into the TTS, run a quick `unicodedata.normalize("NFKC", text)` pass and strip emoji before generation.

### `|spell|` — byte-level fallback for hard words

The tokenizer recognises a special escape: wrap a word in pipes (`|like_this|`) and the bytes inside are passed to the model as raw byte tokens instead of being merged by the BPE tokenizer. During training, ~20% of words were randomly spelled this way (`spell_randomly` in `src/vui/tokenizer.py`), so the model learned to pronounce arbitrary byte sequences character-by-character with the right phonetics — at the cost of slightly less natural prosody on that word.

When to reach for it:

- **Proper nouns the model trips on** — uncommon names, foreign names, surnames. `Saoirse Ronan` often comes out wrong; `|Saoirse| Ronan` gets the "Seersha" pronunciation. `|Buttigieg|`, `|Xochitl|`, `|Siobhan|`, `|Eyjafjallajökull|`.
- **Technical jargon, CLIs, libraries** — `|kubectl|`, `|nginx|`, `|psql|`, `|fzf|`, `|tmux|`. The BPE tokenizer merges these into one or two IDs the model has weak associations for; byte-level forces it to sound out.
- **Brand/product names** — `|Anthropic|`, `|Hugging Face|` (the model usually gets these, but if not).
- **Drug / chemistry / medical terms** — `|acetaminophen|`, `|levothyroxine|`, `|methylphenidate|`. Long Greek/Latin-derived stems are exactly the kind of thing BPE merges badly.
- **Scientific Latin (binomial names)** — `|Quercus| |robur|`, `|Homo| |habilis|`. Wrap each word in its own pair of pipes.
- **Acronyms you want pronounced as a word, not letter-by-letter** — `|SQUID|`, `|FAANG|`, `|SCUBA|`. Without the pipes the model often defaults to spelling them out ("S-C-U-B-A"); piped, they get sounded out as words.
- **Misheard / drifted words** — the safest fix after a bad take is to just `|wrap|` the offending word and re-render.

Syntax rules:

- Pipes must be **adjacent to the word** with no spaces — `|kubectl|`, not `| kubectl |`.
- The regex (`SPELL_PATTERN`) explicitly **excludes `<` and `>`** so it won't collide with the model's `<|time|>` tokens — but stay away from `|<...>|` shapes.
- Only one word per pair of pipes (no inner pipes). For multi-word names, wrap each word: `|Saoirse| |Ronan|`.
- Don't wrap tags (`[breath]`, `[laugh]`) or numbers — tags have their own slot and numbers should be **spelled phonetically** as words (above), then optionally byte-escaped if even the spelled-out form trips the model (`|forty| |ninety|` is rarely needed).

When NOT to use it:

- **Common English words** — the BPE tokenizer already handles them with better prosody. Byte-level is a last resort, not a default.
- **Whole sentences** — wrapping everything in pipes flattens the prosody. Use it surgically, one word at a time.

If you're unsure whether a word needs it, render once without — if the pronunciation's off, add `|pipes|` around that one word and re-render. The fix is almost always that local.

## Quick checklist

Before you ship a clip:

- [ ] Voice prompt audio is **clean** (no background noise, decent mic, 24 kHz if possible).
- [ ] Voice prompt transcript **exactly** matches the audio (filler + tags included).
- [ ] Voice prompt is **≥30 seconds** ideally; longer for distinctive accents.
- [ ] Output text has **no digits, no symbols, no markdown** — everything spelled phonetically.
- [ ] Pauses in the text use **ellipsis or em-dash**, never bare periods mid-clause.
- [ ] Tags are **inline, bracketed, untouched by punctuation**.

If the output sounds wrong, suspect the inputs before the model — 90% of "Vui sounds bad" reports trace back to a mismatched transcript or noisy prompt audio.
