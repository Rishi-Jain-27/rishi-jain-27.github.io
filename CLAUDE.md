# CLAUDE.md

Notes for Claude when working on this site. Living doc — keep it current.

## What this site is

Rishi Jain's personal site + blog. Jekyll, originally Centrarium but heavily customized. Hosted at https://rishi-jain-27.github.io via GitHub Pages. `baseurl` is empty (root hosting).

Optimized for, in order: (1) **readers** of long-form writing; (2) **people evaluating projects** (posts link out via the inline pill); (3) **professional landings** (recruiters reading resume/about). Not a scratchpad — posts get edited and polished.

---

# WRITING VOICE

This is the most important section. It is the source of truth for how Rishi writes. Apply with judgment — spirit over letter. A post that uses three of these tendencies naturally beats one that forces in ten.

Before drafting, read a couple of existing posts to hear the voice.

## The one thing to never get wrong

Two halves, both required:
1. **Hatred of generic, AI-sounding writing.** Overperformance is the cardinal sin — fake enthusiasm, manufactured vulnerability, hype, drama over small things, strained punchiness. Plain, composed, understated. Never strained.
2. **Love of explanation, images, and mild humor.** Thorough mechanism-revealing teaching, visuals that do real work, dry wit at the surface.

Underneath both: **understanding-first, always.** Never write, claim, or skip a step across a gap in genuine understanding. And note what "simple" means here — see below.

## "Simple" means complete, not easy

When Rishi says "simple explanation," he means an **in-depth explanation that covers every detail and everything the reader needs**, made comprehensible. It does NOT mean stripped-down or easy. Simplifying by **omitting detail or giving false information is lying.** Completeness + clarity, never reduction.

## The two modes (decide this first)

1. **Teaching / learning post** (the default; FER2013, LSTM, MNIST, linear-algebra, quantization posts are this): cumulative vocabulary, worked examples, every derivation step, mechanism images, first person, **dry humor allowed at the surface.**
2. **Formal findings / project documentation:** a brief first-person narrative frame, then **professional — no humor, no fluff, get to the point.**

Trigger: am I learning-and-teaching (humor on) or documenting formal findings (humor off)? Pick the mode, then write.

Also: **Rishi's chat/speech voice ≠ his blog voice.** Casual register ("kinda," "lowkey") is chat-only. The blog is the composed British register, always.

## Register, rhythm, sentences

- **Composed, British-reminiscent register.** Refinement, taste, correctness. He edits by ear — a sentence can be logically correct and still wrong because it sounds off.
- **Long, winding, controlled sentences.** He dislikes short clipped sentences. If a sentence gets *too* long it's a clarity failure — rewrite for simplicity, don't just chop. The enemy isn't length, it's a sentence long because it wasn't understood well enough to say cleanly.
- **No one-sentence paragraphs** (too staccato). Paragraphs are **idea-based** — as long as the idea needs, a new idea starts a new paragraph. No fixed length.
- Vary rhythm naturally, but trend long and flowing. **No staccato, no fragments, no "Not X. Y." punch.**

## Punctuation & formatting

- **Never em-dashes.** Anywhere. A **colon** does the em-dash's job. Hyphens in compounds (`first-person`, `64×64`) and ranges (`600–1000`) are fine; the long dash is the thing to avoid.
- **Never semicolons.**
- **Never bold, never italicize.** Emphasis comes only from sentence structure and word choice — no typographic shouting.
- **No bullet points** — they're a cop-out that skips the connective sentences that *are* the writing. Numbered lists only rarely, for a true sequence; even then prefer inline "First,... Then,...".
- **No emojis** (zero tolerance). Exclamation points only sarcastically, once in a blue moon.
- Digits over spelled-out numbers (`3 layers`, `64×64`). Consistent notation throughout — sloppy/inconsistent notation is a clarity failure.
- Footnotes are good. Sparing parenthetical asides, never overdone.

## Structure of a post

- **Openings** (a known soft spot): narrative/context, not a cold thesis. "The other day I was..." or "I wanted to learn more about {x}, so..." is the comfort default. At most a small hook — never force the reader, never glamorize with fancy quotes.
- **Order:** intuition first, then the formal definition. Concept first, then hands-on — interleave when integration makes it clearer.
- **Headings name the topic, never the claim** — the claim lives in the paragraph beneath. Sentence case.
- **Declare the assumed-knowledge contract** up front; a prerequisites line especially when a post builds on a previous one.
- **Recaps** ("so far we've seen...") are useful signposting in long posts — keep them.
- **Closings:** stop when done. A `## Takeaway` (or `## What's Next` for project posts) only if there genuinely is one — don't manufacture a bow. The last sentence tends to be a clean technical claim or a look-ahead. No aphorism.
- **Length:** no hard cap — write what the explanation needs; completeness beats a word count. When trimming, **repetition and tangents go first; the explanation is the last thing standing.** No padding either — quality over length and over brevity.

## Technical content (required on technical topics)

A technical post **needs the code, the images, and the math.** Pure prose isn't enough.

- **Code:** prose explanation first (build understanding) → then the block → then a line-by-line walkthrough only for genuinely dense parts. Show **surgical** code — just the relevant lines tied to the prose; whole function only when context requires. Explanation lives in the prose; in-code comments only mark section boundaries, never line-by-line narration.
- **Math:** show **every step.** "After some algebra..." is unacceptable — it hides the step you maybe couldn't do. Heavy notation is fine (precision matters) but every part of hard notation must be explained. Verify math with Claude. Personal voice stays *out* of derivations (clean impersonal "we"). See the MathJax/kramdown rules under Conventions.
- **Worked examples:** tiny toy numbers you can verify in your head, so the mechanism is visible and nothing hides behind arithmetic.
- **Visualizations are not optional.** A visual must **show the mechanism**; a static diagram earns its place only when it reveals structure that's hard to picture otherwise. No decorative/AI-generated hero images. Hierarchy: **images > analogies > plain prose.** Analogies are useful but must not mislead.
- Understand from scratch (implement if useful), *then* reach for the library. The non-negotiable is understanding, not reimplementation.

## Humor & personality

- **Dry, British understatement** is the humor and the memorability engine. Loved words: "a bit," "rather," "quite," "not exactly," "probably less than ideal." British tag questions: "...wasn't it?". The gentle inclusive "let's" ("let's break it down").
- The butt of the humor is the **subject**, never himself. Others aren't mocked — they're silently critiqued by doing the harder, more honest thing (state it flatly, no spin).
- **Not self-deprecating** — state what you did flatly, without a biasing factor up or down; the dry gap reads as humor. "It was a bit difficult — here's how I did it," then pivot to teaching. No dwelling on struggle, no performed vulnerability.
- Excited and skeptical both stay understated. Enthusiasm ceiling: "elegant," "rather lovely," "a bit beautiful," "sweet." Skeptical: "I'm not too sure about...".
- **Humor switches off in the deep, serious, complex parts.** The harder the material, the more sober the voice.
- No personification of models. No profanity (clashes with the register); use "yikes" instead.

## Honesty (the spine)

- **Uncertainty:** hedge genuinely uncertain things with "probably / maybe / perhaps"; verify with sources until it's gone, or explicitly flag it and invite readers to email. But **commit where you know** — never-certain-about-anything is spineless mush.
- **State limitations** — they make a post stronger and more honest. Their absence is a tell.
- **Corrections are silent edits** — the blog is a teaching artifact; its job is correct understanding, not a changelog.
- Cite papers with a **link, not author names** (name-dropping borrows authority). Plain descriptive titles — no brag. A subtitle only when the title isn't descriptive enough.
- Disagreement is handled by **correcting the record**, naming the source obliquely as reader-service ("beware of sources that botch the math here, like X"), never a pure takedown.

## Banned: AI tells

If even one appears, the writing fails. These don't conflict with the voice above — both reject machine-flavored prose.

**Dead AI vocabulary** (statistically overrepresented in LLM output — the fingerprint): delve, realm, harness, unlock, tapestry, paradigm, cutting-edge, revolutionize, landscape (abstract), intricate/intricacies, showcasing, crucial, pivotal, surpass, meticulously, vibrant, unparalleled, underscore, leverage, synergy, innovative, game-changer, testament, commendable, highlight (verb), emphasize, boast, groundbreaking, align, foster, showcase, enhance, holistic, garner, accentuate, pioneering, unleash, versatile, transformative, redefine, seamless, optimize, scalable, robust, breakthrough, empower, streamline, frictionless, elevate, adaptive, effortless, data-driven, insightful, proactive, visionary, disruptive, reimagine, unprecedented, intuitive, democratize, accelerate, state-of-the-art, dynamic, immersive, predictive, supercharge, interplay, captivate. Also: "serves as / stands as / marks a / represents a / boasts a / features a / offers a" to dodge "is" or "has" — just say "is."

**Dead phrases:** "In today's [anything]...", "It's important to note that / It's worth noting", "In order to" (say "to"), "Let's dive in / explore / unpack / delve into", "At the end of the day", "Moving forward", "To put this in perspective", "What makes this particularly interesting is", "In other words", "It goes without saying", "Here's what nobody tells you" / anything with "nobody" or "most people don't realize", "In this article I will..." (announcing what you're about to do).

**Dead transitions:** "Furthermore / Additionally / Moreover", "That said / That being said", "With that in mind", "On top of that", any mechanical college-essay connector.

**Engagement bait:** "Let that sink in / Read that again / Full stop", "This changes everything", "Are you paying attention?".

**Hype:** "Supercharge / Unlock / Future-proof", "10x your [anything]", "game-changer / cutting-edge", any promise of superpowers or overnight transformation.

**THE BIG ONE (fatal): negative parallelisms / reframe constructions.** The single most reliable AI tell. If you see one, rewrite the whole sentence.
- "This isn't X. This is Y." / "Not X. Y." / "Forget X, this is Y." / "Less X, more Y." / "Not only X, but also Y." / "It's not just about X, it's about Y." / "X is dead. Y is the future." / "The question isn't X, it's Y." / "You don't need X. You need Y." / "X is overrated. Y is what matters." / ANY sentence that negates one framing then asserts a corrected one.
- Sneaky versions: "While X might seem right, Y is actually...", "Sure, X works. But Y is where the real...", "X gets all the attention, but Y is what actually...".
- Fix: delete everything before the positive claim. "It's not about the prompt, it's about the context" → "It's about the context."

**Other AI patterns:** puffery / significance inflation ("a pivotal moment", "marking a significant shift") — state the fact, let the reader judge. Rule of three ("speed, efficiency, and innovation") — use two or four, or just the one that matters. False ranges ("from ancient traditions to modern innovations"). Elegant variation (swapping a name for "the protagonist" / "the key player" — just reuse the name). Meta-commentary ("in this section we will...") — say the thing; a *little* reassuring framing is fine ("this looks complicated, but it's not really — let's break it down"). Fake-depth participles ("highlighting its importance", "underscoring its significance"). Knowledge-cutoff disclaimers. Chat leakage ("I hope this helps!", "Certainly!", "Great question!"). Title Case In Headers (use sentence case). Hollow intensifiers as filler (*actually*, *literally*, *honestly*, *essentially*) — fine only when they carry real meaning.

Also banned hooks/structures Rishi specifically rejects: clickbait curiosity-gap intros, fake-vulnerability hooks, listicles, Twitter-style one-line paragraphs (the apex crime — egregiously AI and overperforming), "Yes, that's right:", "obviously / clearly / trivially / it's easy" (makes a struggling reader feel dumb), closing aphorisms and neat-bow closers ("That stuck with me."), reader-instruction asides ("Read that twice").

## What makes a piece distrusted (apply to your own drafts)

Buzzwords, no code, no math, no intuitive explanation, salesy tone. Shallow understanding shows as: no "why," no connection to broader topics, writing that hinges on reader intuition instead of doing the work, describing *what* without *how* or *why*. The sharpest fakery detector is **a missing worked example.** (Note: this bar is for explainer blogs — research papers are allowed to be dense because their job is different. Judge a text against its own purpose. Don't reflexively distrust a high reported result; investigate before judging.)

## Litmus test

> Does this sound like something Rishi would actually write, or like an AI trying very hard to imitate him?

If it feels forced or performed, pull back. Less imitation, more inhabitation.

---

## Style direction (visual / design)

**Dark + warm-gold, slim, code-flavored.** Keep this base. Propose small tweaks (palette, type, spacing, motion) when they fit; don't redesign without asking.

Personality the site projects: **warm/inviting + bold/opinionated** — personal, not a sterile engineer-portfolio; confident enough to have a point of view. Avoid the "safe minimal blog template" feel.

Motion appetite: **add more, tastefully.** Existing motion is the floor. Subtle hover micro-interactions, page transitions, decorative motion welcome — all respecting `prefers-reduced-motion`. No flashy/distracting motion.

### Theme tokens (in [css/main.scss](css/main.scss))

```
$bg        #1e1f2a   dark navy-charcoal
$bg-elev   #292a36   one step up (cards, code, chips)
$fg        #f0f0f5   primary text
$fg-muted  #adadba   secondary text
$fg-dim    #797a85   tertiary (dates, meta)
$rule      #363745   borders / dividers
$accent    #d4af7a   warm gold, THE color
$accent-hi #ebc999   hover/lifted accent
```

Background: two soft radial `$accent` gradients over `$bg` for ambient glow, on a dedicated fixed `.bg-glow` layer (not `body`) so it can drift via a compositor-only transform; `body` is solid `$bg`.

### Typography

- **Sans:** system stack, Inter preferred. Body weight `300`.
- **Mono:** `ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, ...`. Used for accents, section markers, project pills, tech chips, code.
- Headings: weight 500–600, letter-spacing slightly negative.

### Recurring visual signatures

- **`// 01` section markers**, auto-numbered via CSS counters on `.page/.post .post-content h2`. NOTE: Rishi wants to move off the `//` code-comment style toward a dash or light numbering — propose this when touching heading styles.
- **Accent left-bar slide-in** on post-list hover (`.post-list > li::before`); hover nudges the item right 0.7rem.
- **Inline project pill** next to post titles when frontmatter has `project_url` (uppercase mono badge with `↗`).
- **Accent diamond divider** (`◆`) instead of a horizontal rule.
- **Tech-tile grid** on the home hero (monochrome simpleicons CDN icons via CSS mask + `--icon-color`; non-iconed tiles use a glyph).
- **64px accent underline** at the header's bottom-left (`.site-header-container::after`).

## Repo layout

All styling lives in `css/main.scss` — there is no `_sass/` directory (the dead Centrarium `_sass/` light-theme, archive layout, and tooltips were deleted in 2026-05; `main.scss` never imported them). To add partials you'd re-create `_sass/` and `@import` from `main.scss`.

```
_config.yml             title, social links, OG image, pagination (5/page), exclude list
_layouts/
  default.html          shell: head + header + {{ content }} + footer; adds .bg-glow
  page.html             static pages (resume, about, posts index)
  post.html             blog posts; adds .reading-progress
_includes/
  head.html             <head>, meta, favicons, OG; h2 scroll-reveal observer; Motion choreography; scroll-spy rail builder
  header.html           top nav: .logo + .navigation-menu + ⌘K trigger
  nav_links.html        iterates site.pages where main_nav: true
  footer.html           three-column footer: nav / contact / signature
  command_palette.html  ⌘K palette: markup + Liquid-built index + behaviour
  viz/vectors.html      reusable interactive 2D vector widget
  page_divider.html     the ◆ divider
_posts/                 YYYY-MM-DD-slug.md
css/main.scss           ★ the only stylesheet, self-contained (no @import)
js/                     empty; JS lives inline in head.html (Motion from CDN)
index.html              home: hero (photo + title + tech grid) + paginated post list
posts.md / resume.md / about.md   nav pages
assets/                 profile-hero.png, logo.png, lin-alg-vectors-3d.png, /icons/
feed.xml                RSS
```

---

## Conventions

### Adding a blog post

`_posts/YYYY-MM-DD-slug.md` with frontmatter:

```yaml
---
layout: post
title: "Title goes here"
author: "Rishi Jain"
date: 2026-05-03
categories: projects        # used on /posts/ grouping
project_url: https://example.com    # optional, adds inline pill
project_label: "Visit"               # optional, defaults to "Open"
math: true                           # optional, loads MathJax
---
```

Body is kramdown. `## Section` headings auto-get `// 01`, `// 02` markers. Follow the WRITING VOICE section above. A project post should also (1) carry `project_url`/`project_label` and (2) link the project early in the body.

### Adding a nav page

Top-level `.md` with `layout: page`, a `permalink:`, and `main_nav: true` (the flag [_includes/nav_links.html](_includes/nav_links.html) iterates on).

### Math (LaTeX via MathJax)

MathJax v3 from CDN, **opt-in per page** via `math: true`. Pages without it don't fetch MathJax.

- **Inline:** `\\( ... \\)`. The doubled backslashes are required — kramdown's escape list includes `( ) [ ] | ! { }`, so single-backslash `\(` is stripped before MathJax sees it. `\\(` in source → `\(` in HTML.
- **Display:** `\\[ ... \\]` on its own lines.
- Inside math, only the *delimiter* backslashes need doubling. `\frac`, `\partial`, `\sum`, etc. are safe single-backslash. Double these inside math: `\(`, `\)`, `\[`, `\]`, `\{`, `\}`, `\|`, `\!`, `\_`, `\#`, `\*`, `\-`, `\+`, `\.`, `\~`, `\=`, `\"`, `\'`, `\<`, `\>`, and backtick.
- **Bare underscores at word boundaries get eaten as emphasis.** Intra-word `_` is fine (`y_t`, `h_t`, `\sigma_1`). But a closing brace followed by `_` (`}_t`, `}_{...}`, `)_{`) opens/closes kramdown emphasis and consumes the underscores (common with `\underbrace{...}_{...}`). Fix by escaping: write `}\_t`, `}\_{...}`. The `\_` survives as a literal `_` for MathJax.
- **Do not** use `$ ... $` (collides with currency) or `$$ ... $$` (kramdown math is disabled in [_config.yml](_config.yml)); use `\\[ \\]`.
- MathJax skips `pre`/`code`, so backtick samples render literally. Equation numbering is off (`tags: 'none'` in [_includes/head.html](_includes/head.html)). Display equations scroll horizontally on overflow (the `mjx-container[display="true"]` rule in main.scss).

### Motion / animation (two systems, both in [_includes/head.html](_includes/head.html))

Both gated on `prefers-reduced-motion: no-preference`, both no-ops if their JS doesn't run.

1. **IntersectionObserver (dependency-free):** adds `.js-reveal` to `<html>`, observes in-article `.page/.post .post-content > h2` for scroll fade-in (CSS in the `.js-reveal …` block).
2. **Motion ([motion.dev](https://motion.dev), pinned `@11.11.13`, don't use `@latest`):** dynamic `import(...)` adds `.js-motion` to `<html>` (CSS hides animated elements so there's no flash), then springs in: page-load `.page-content`; home hero (`.hero-photo-frame` → title/subtitle → staggered `.tech-tile`); post-list items via `inView`; hero-photo parallax; reading-progress bar `scaleX`; magnetic hover on `.tech-tile`/`.project-button` (fine-pointer only).

A JS-free third piece: ambient gold glow (`.bg-glow`) drifts via CSS `@keyframes glow-drift`.

Gotchas:
- **Fallback safety:** a 2.5s fallback (and `.catch`) removes `.js-motion`, reverting to the visible resting state if the CDN never loads. Reduced-motion users bail before import. Never leave an element hidden only by JS with no fallback.
- **Inline-style cleanup:** Motion leaves inline `transform`/`opacity`, which beat `:hover` rules. The tech-tile cascade strips its inline transforms on completion (and re-pins `opacity:1`) so `:hover` keeps working. Do the same for any animated element that also has a CSS hover/active transform.
- Add new **entrance** targets to the `.js-motion …` hide list in main.scss (or they flash). **Don't** add ongoing-interaction targets (parallax, magnetic hover, progress bar) — they must be visible at rest.

### Interactive components (vanilla JS, none depend on the Motion CDN)

- **⌘K command palette** ([_includes/command_palette.html](_includes/command_palette.html), `.cmdk*`, included once in default.html): opens on ⌘K/Ctrl-K, `/` (unless typing in a field), or the `[data-cmdk-open]` trigger. Index is **Liquid-built at build time** (all posts, `main_nav` pages, static actions) as a JS array literal. Fuzzy subsequence match over title+categories; external actions (`"x": true`) open in a new tab. Add a quick action by appending to `INDEX`.
- **Scroll-spy section rail** (second `<script>` in head.html, `.section-rail*`): on posts with ≥2 `##` sections, builds a fixed left rail of the numbered markers; active section lights gold (IntersectionObserver), hover expands labels, click smooth-scrolls (honouring reduced-motion). Hidden below 1240px. Assigns `id`s (`section-1`, …) to h2s lacking them.
- **Interactive vector widget** ([_includes/viz/vectors.html](_includes/viz/vectors.html), `.vec-viz*`): canvas with two draggable vectors; live dot product, norms, cosine, angle. Drop in with `{% raw %}{% include viz/vectors.html %}{% endraw %}` (optional `a="4,1" b="1,3"`, grid ±6, snap 0.5). Init ships only on pages using the include (guarded by `window.__vecVizBooted`), scans all `.vec-viz`. Theme colours are hardcoded as a `C` map mirroring the SCSS tokens — update both if the palette changes. It's 2D, so on-page examples should use 2D vectors.

### Tech tiles on home (inline in [index.html](index.html))

- **With icon:** `<a class="tech-tile">` + `<span class="tech-icon" style="--icon-url: url('https://cdn.simpleicons.org/<name>'); --icon-color: #HEX;">`.
- **Without icon:** `<a class="tech-tile tech-tile--noicon">` + `<span class="tech-icon-text">` with a glyph (`{ }`, `▦`, `✦`, `⚠`).

Keep the grid curated — technologies Rishi actually uses, not a logo wall.

## Source material workflow

`source-material/` is the local-only drop zone for raw input for upcoming posts (gitignored except `.gitkeep`; in `_config.yml` `exclude:`). When asked to write about something there: list the dir, read the files, draft the post in `_posts/` per the voice above, then **ask before deleting** the source files (don't delete until Rishi confirms). If multiple unrelated items are present, ask which one.

## Local development

```
bundle install
bundle exec jekyll serve   # → http://localhost:4000
```

`_site/` is build output — gitignored, regenerated by GitHub Pages; don't hand-edit.

**Local preview is currently broken on this machine** (2026-05-16): the `github-pages` gem pins old C-extension deps that won't compile against macOS Tahoe's toolchain. Rishi has accepted skipping local preview and relying on GitHub Pages' own build. Don't yak-shave on it; if asked to fix it, treat it as a real setup task and surface cost/benefit first.

## Commit style

**Never `git commit` or `git push` yourself.** Make the file changes, then stop and let Rishi review and commit/push. Even when he says "commit and push," prepare the change and hand it back rather than running the git commands.

Short, present-tense, action-first (not Conventional Commits). E.g. "Add scroll-reveal fade-up for post-list items and h2 sections". One line where possible. Look at recent commits before writing one.

## Things to be careful with

- **Don't reintroduce Font Awesome / Bootstrap** — dropped deliberately (CSS zeroes out leftover `.fa-stack` / `.social-media-list i`).
- **Don't widen the content column** without asking — `$content-width: 640px` is deliberate.
- **Respect `prefers-reduced-motion`** — wrap new motion in `@media (prefers-reduced-motion: no-preference)`.
- **All styling goes in `css/main.scss`** — no `_sass/` tree.
- **`baseurl` is empty** — don't add one unless hosting changes.
- **In prose, never bold/italicize and never use em-dashes** — see WRITING VOICE.
