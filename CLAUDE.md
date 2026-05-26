# CLAUDE.md

Notes for Claude when working on this site. Living doc, keep it up to date as the site evolves.

## What this site is

Rishi Jain's personal site + blog. Jekyll, originally built on the Centrarium theme but heavily customized. Hosted at https://rishi-jain-27.github.io via GitHub Pages.

Optimized for three audiences, in roughly this order:
1. **Readers** of long-form writing / blog posts.
2. **People evaluating projects**, each post can link out to a live project via the inline pill button.
3. **Professional landings**, recruiters and collaborators reading the resume / about page.

Not a casual scratchpad. Posts get edited and polished.

## Style direction

Current direction: **dark + warm-gold, slim, code-flavored.** Keep this base. Refinements welcome, propose small tweaks (palette nudges, type, spacing, motion) when they fit, but don't redesign without asking.

Personality the site should project: **warm/inviting + bold/opinionated.** Soft enough to feel personal (not a sterile engineer-portfolio), confident enough in type, color, and voice to feel like it has a point of view. Avoid the "safe minimal blog template" feel.

Motion appetite: **add more, tastefully.** Existing motion (scroll-reveal fade-up, hero photo pulse, accent left-bar slide on post-list hover) is the floor, not the ceiling. Subtle hover micro-interactions, page transitions, decorative motion are welcome as long as they respect `prefers-reduced-motion`. No flashy/distracting motion.

### Theme tokens (defined in [css/main.scss](css/main.scss))

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

Background uses two soft radial gradients of `$accent` over `$bg` for ambient glow; `background-attachment: fixed`.

### Typography

- **Sans:** system stack with Inter preferred (`-apple-system, BlinkMacSystemFont, "Inter", ...`). Body weight `300`.
- **Mono:** `ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, ...`. Used for **accents**, section markers, project pills, tech chips, code.
- Headings: weight 500–600, letter-spacing slightly negative (`-0.01em` / `-0.02em`).
- Note: `_sass/base/_variables.scss` defines an *older* light-theme (Open Sans / Roboto Slab), that's legacy from the Centrarium theme and is overridden by [css/main.scss](css/main.scss). Don't edit the old vars expecting visible change.

### Recurring visual signatures (preserve these)

- **`// 01` section markers**, auto-numbered via CSS counters on `.page .post-content h2` and `.post .post-content h2`. The `//` makes them feel like code comments.
- **Accent left-bar slide-in** on post-list hover (`.post-list > li::before`). Hover also nudges the item right by 0.7rem.
- **Inline project pill** next to post titles when frontmatter has `project_url`, small uppercase mono badge with `↗`. See "Posts" below.
- **Slim accent diamond divider** (`◆`) instead of the original Centrarium horizontal rule.
- **Tech-tile grid** on the home hero, monochrome simpleicons CDN icons painted via CSS mask + `--icon-color`. Non-iconed tiles use a text glyph (e.g. `{ }`, `▦`, `✦`).
- **64px accent underline** at the bottom-left of the site/post header (`.site-header-container::after`).

## Repo layout

Jekyll defaults plus a few quirks. Most "real" styling lives in `css/main.scss`, **not** in `_sass/`.

```
_config.yml             site title, social links, OG/twitter image, pagination (5/page)
_layouts/
  default.html          shell: head + header + {{ content }} + footer
  page.html             static pages (resume, about, posts index)
  post.html             blog posts
  archive.html          (Centrarium legacy, not actively used)
_includes/
  head.html             <head>, meta, favicons, OG, the h2 scroll-reveal IntersectionObserver, AND the Motion entrance choreography
  header.html           top nav with .logo + .navigation-menu (populated from nav_links.html)
  nav_links.html        iterates site.pages where main_nav: true
  footer.html           three-column footer: nav / contact / signature
  page_divider.html     the ◆ divider
_posts/                 YYYY-MM-DD-slug.md, blog content
_sass/                  Centrarium legacy SCSS (mostly overridden, see note)
css/main.scss           ★ the real stylesheet. ~690 lines, defines theme tokens.
js/                     empty, JS lives inline in _includes/head.html (Motion is loaded from CDN there)
index.html              home: hero (photo + title + tech grid) + paginated post list
posts.md                /posts/, posts grouped by category
resume.md               /resume/, long markdown resume
about.md                /about/, short bio
assets/                 profile-hero.png, header_image.jpg, logo.png, /icons/ favicons
feed.xml                RSS
Gemfile / Gemfile.lock  github-pages gem
```

### About `_sass/`

`_sass/base/_variables.scss`, `_layout.scss`, `bourbon/`, `neat/` are all **inherited from the Centrarium theme**. They define a light-mode palette and the old layout. `css/main.scss` mostly overrides all of it (and even hides leftover Font Awesome icons it can't suppress). **Default to editing `css/main.scss`**, not `_sass/`. If you find yourself wanting to fix something in `_sass/`, first check whether `main.scss` already overrides it, usually it does.

## Conventions

### Adding a blog post

Create `_posts/YYYY-MM-DD-slug.md` with frontmatter:

```yaml
---
layout: post
title: "Title goes here"
author: "Rishi Jain"
date: 2026-05-03
categories: projects        # used on /posts/ grouping
project_url: https://example.com    # optional, adds inline pill
project_label: "Visit"               # optional, defaults to "Open"
math: true                           # optional, loads MathJax for LaTeX (see "Math" below)
---
```

Body is kramdown. Use `## Section` for top-level sections (will auto-get `// 01`, `// 02` markers). Use `**bold**` sparingly, bold gets `$fg` (brighter) so it pops in the muted body text.

#### Writing voice (read the existing posts before drafting a new one)

The published posts have a consistent voice. Match it. The voice was deliberately tightened in 2026-05 to cut AI-flavored prose tells; everything in this section is the post-tightening direction.

- **Title:** lowercase-after-the-first-word, no em-dash subtitle, no "X Was the Y" / "X — When Y" subtitle formulas. A colon is allowed but use it very sparingly (at most one post in a batch). Avoid generic "Building X" / "How I Built X" titles too. Aim for a short noun-phrase title that says what the post is about. ("Transfer learning lost to my small CNN on FER2013", "A first LSTM in PyTorch, classifying MNIST as a sequence".)
- **Open with the thesis or the surprising result**, not setup or background. The first paragraph usually drops the GitHub link and any cross-references to related posts.
- **Section headings:** sentence-case, claim-bearing where natural, but don't strain to make every heading a slogan. "Why the bigger model lost" is fine. "The three things that go wrong" is fine. "Architecture bake-off" is also fine.
- **Self-aware about clichés.** Posts often name the conventional wisdom they're pushing against ("tutorials are a trap", "always start with transfer learning", "MNIST is solved") and answer it.
- **Specific over abstract.** Name hyperparameters, file paths, library versions, exact accuracy numbers. Concrete details over general principles.
- **Honest about failures.** First-person, mildly self-deprecating, never preachy or didactic. Avoid the constructed-feeling self-deprecation move where the writer announces their own surprise as a structural beat ("I was surprised for about an afternoon, and then it made sense"). Say what happened, not how you felt about it for narrative effect.
- **Math and technical depth are welcome, and expected on technical topics.** The personal-narrative voice is texture, not substance. If a post is about a technical subject (an algorithm, an architecture, a derivation), it should do the technical work: write the equation, derive the gradient, work the concrete example, name the algorithm with enough precision that a reader could reimplement it. Trials and struggles are welcome alongside, not as a substitute.
- **End with a short `## Takeaway`** (or `## What's Next` for project posts with a roadmap). Two or three sentences. A final claim, not a restated summary, and *not* a neat AI-style aphorism ("X isn't Y. Y is Z." / "Bigger isn't smarter. Closer is."). Just say the last thing and stop.
- **Post length:** narrative-only posts target roughly 600–1000 words (cut past 1200). Technical posts with derivations can run longer, up to about 2000 words if the math earns it. Cut filler, never cut the math.
- No emojis. No "I hope this helps." No advice paragraphs aimed at the reader.

##### AI-tell list (avoid)

These patterns are how AI-written prose gives itself away. They were all stripped from the existing posts in 2026-05. Don't reintroduce them.

- **Em-dashes (`—`).** Do not use em-dashes anywhere: not in titles, not mid-sentence, not in bullet lists ("`- **Status** — answered`"). Use commas, parentheses, periods, or just restructure. Hyphens in compound words (`first-person`, `64×64`) are obviously fine; ranges like `600–1000` are fine. The character to avoid is the long dash.
- **"That stuck with me." / "That mattered." / "That's the [X]." closers.** Sentences that exist only to put a neat bow on the preceding paragraph. Cut them.
- **"X is Y. Y is Z. Z is W." parallel rhythms,** especially as a paragraph-closing flourish. AI loves the cascade. Vary sentence length and structure instead.
- **"Not X. Y." emphatic fragments** ("The model goes in the trash; the infrastructure ships." "Bigger isn't smarter. Closer is." "Useful as a warm-up. Not useful as an application.") Cut or merge into a normal sentence.
- **Italicized single-word emphasis for rhythm** (`*the*`, `*actually*`, `*gone*`). Italics are fine for genuine technical emphasis (e.g. the *same* W to stress identity), terrible for prose drama.
- **"It's the kind of work that..." / "It's the difference between..." / "What separates X from Y is..."** writerly hedges. Drop.
- **Closing aphorisms.** "The problem is the product. Everything else is engineering around it." "Bigger isn't smarter. Closer is." Don't write summary-mottos. The takeaway can be a normal paragraph.
- **Reader-instruction asides.** "Read that twice." / "Internalize this and X..." / "Notice that..." Don't address the reader's mental state.
- **Hollow intensifiers.** *actually*, *literally*, *honestly*, *essentially*. Allowed when they carry real meaning (e.g. "literally the same vector" when two things are bit-identical, "making it actually run" contrasting with theoretical), banned as filler.

### Adding a nav page

Add a top-level `.md` (like [about.md](about.md), [posts.md](posts.md), [resume.md](resume.md)) with:

```yaml
---
layout: page
title: "Title"
permalink: /your-path/
main_nav: true     # this is what puts it in the nav
---
```

`main_nav: true` is the magic, [_includes/nav_links.html](_includes/nav_links.html) iterates `site.pages` and only includes pages with this flag.

### Math (LaTeX via MathJax)

The site loads MathJax v3 from a CDN, **opt-in per page** via `math: true` in frontmatter. Pages without that flag don't fetch MathJax at all.

- **Inline math:** `\\( ... \\)`, e.g. `\\( h_t = f(W h_{t-1} + U x_t) \\)`. The doubled backslashes are required: kramdown's backslash-escape list includes `( ) [ ] | ! { }`, so a single-backslash `\(` gets stripped to a bare `(` before MathJax can see it. `\\(` in source becomes `\(` in HTML, which is what MathJax matches against.
- **Display math:** `\\[ ... \\]` on its own line(s), e.g.:
  ```
  \\[
  \frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
  \\]
  ```
- **Inside the math**, only the *delimiter* backslashes need doubling. LaTeX commands like `\frac`, `\partial`, `\sum`, `\delta`, `\,`, `\qquad` are safe as single-backslash because the character that follows isn't in kramdown's escape list. The ones to watch for and double inside math are `\(`, `\)`, `\[`, `\]`, `\{`, `\}`, `\|`, `\!`, `\_`, `\#`, `\*`, `\-`, `\+`, `\.`, `\~`, `\=`, `\"`, `\'`, `\<`, `\>`, `\` `` ` ``.
- **Bare underscores at word boundaries get eaten as emphasis.** Intra-word `_` is fine (kramdown's emphasis rule explicitly skips alnum-`_`-alnum, so `y_t`, `h_t`, `W_l`, `\sigma_1` all work). But `}_t` (non-alnum to the left, alnum to the right) opens emphasis, and `\sum_{` (alnum to the left, non-alnum to the right) closes it. If both appear in the same math block, kramdown wraps everything between in `<em>...</em>` and consumes both underscores. Also dangerous: `}_{` and `)_{` (both sides non-alnum), these can act as either open or close, and pair up with each other (common with `\underbrace{...}_{...}`, `\overbrace{...}^{...}`-adjacent patterns, anything where a closing brace is immediately followed by `_{`). Fix by escaping the underscore: write `}\_t` and `}\_{...}`. The `\_` survives kramdown as a literal `_` for MathJax to read.
- **Do not use `$ ... $` for inline** (collides with currency in prose) and **do not use `$$ ... $$`** (kramdown's math handling is disabled in [_config.yml](_config.yml), but `\\[ \\]` is the convention here regardless).
- MathJax is configured to skip `pre` and `code` blocks, so backtick code samples render literally, useful for showing the LaTeX source itself.
- Equation numbering is off (`tags: 'none'`). If a future post needs numbered equations, flip that in [_includes/head.html](_includes/head.html).
- Display equations get a horizontal scrollbar on overflow rather than blowing past the 640px content column, see the `mjx-container[display="true"]` rule in [css/main.scss](css/main.scss).

### Motion / animation (two systems, both in [_includes/head.html](_includes/head.html))

There are now **two** entrance systems, both gated on `prefers-reduced-motion: no-preference` and both no-ops if their JS doesn't run:

**1. IntersectionObserver (dependency-free), in-article h2 markers only.**
Adds `.js-reveal` to `<html>` and observes:
- `.page .post-content > h2`
- `.post .post-content > h2`

CSS for it lives in the `.js-reveal …` block in [css/main.scss](css/main.scss). Use this for content that should fade in on scroll without pulling in Motion.

**2. Motion ([motion.dev](https://motion.dev)), spring-based load/scroll choreography.**
Loaded from CDN via dynamic `import('https://cdn.jsdelivr.net/npm/motion@11.11.13/+esm')` (version pinned; bump deliberately, don't use `@latest`). The bootstrap adds `.js-motion` to `<html>`, which hides the animated elements via CSS (the `.js-motion …` block in [css/main.scss](css/main.scss)) so there's no flash; Motion then springs them in. It drives:
- **Page-load transition**, `.page-content` fades + rises on every page.
- **Home hero**, `.hero-photo-frame`, then `.hero-text .title/.subtitle` + `.hero-stack-label`, then a staggered cascade of `.tech-tile`.
- **Post list**, `.post-list > li` spring in via Motion's `inView` (one-shot) as they scroll into view. (These used to be on the IntersectionObserver above; they moved here.)

Gotchas to respect:
- **Failure / reduced-motion safety:** a 2.5s fallback (and the `.catch`) removes `.js-motion`, reverting everything to its visible resting state if the CDN module never loads. Reduced-motion users bail before the import, Motion is never even fetched. Don't break this; never leave an element hidden only by JS with no fallback.
- **Inline-style cleanup:** Motion leaves inline `transform`/`opacity` on the elements it animates, and inline styles beat stylesheet `:hover` rules. The tech-tile cascade therefore strips its inline transforms on completion so `.tech-tile:hover { transform: … }` keeps working. If you animate any element that also has a CSS hover/active transform, do the same cleanup (and re-pin `opacity:1` rather than clearing it, since the `.js-motion` rule still sets `opacity:0`).
- Add new Motion targets to the `.js-motion …` hide list in [css/main.scss](css/main.scss) too, or they'll flash before animating.

Intensity is tuned to "confident & noticeable" (visible spring bounce, longer stagger). Dial `bounce`/`visualDuration`/`stagger` in [_includes/head.html](_includes/head.html) to taste.

### Tech tiles on home

Defined inline in [index.html](index.html). Two flavors:
- **With icon:** `<a class="tech-tile">` with `<span class="tech-icon" style="--icon-url: url('https://cdn.simpleicons.org/<name>'); --icon-color: #HEX;">`. Icon comes from simpleicons.org CDN as an SVG, painted via CSS mask.
- **Without icon:** `<a class="tech-tile tech-tile--noicon">` with `<span class="tech-icon-text">` containing a glyph / text (e.g. `{ }`, `▦`, `✦`, `⚠`).

Keep the grid feeling curated, these should be technologies Rishi actually uses, not a "look how many logos I can fit" wall.

### Project linking

A blog post about a project should:
1. Add `project_url:` (and optionally `project_label:`) to frontmatter, adds the inline pill to post listings on the home page and `/posts/`.
2. Link to the project inside the post body too (early, usually in the intro paragraph).

## Source material workflow

`source-material/` is the drop zone for raw input for upcoming blog posts. The user puts things there (notes, transcripts, screenshots, READMEs, exported chat logs, links in a `.txt`, etc.); Claude reads them, drafts the post, and then the source files get deleted once the post is in `_posts/`.

- The directory is **gitignored** (except for `.gitkeep`), material stays local and never lands in commits.
- The directory is in `_config.yml` `exclude:`, Jekyll won't try to process it.
- When the user asks to write about something in `source-material/`, list the directory, read the files, draft the post in `_posts/` following the conventions above, and then **ask before deleting** the source files. Do not delete until the user has reviewed the draft and confirmed.
- If multiple unrelated items are in `source-material/` at once, ask which one to write about rather than guessing.

## Local development

```
bundle install        # one time
bundle exec jekyll serve
# → http://localhost:4000
```

`_site/` is the build output. It's tracked but should not be hand-edited (Jekyll regenerates it).

**Heads up, local preview is currently not working on this machine** (as of 2026-05-16). The `github-pages` gem pins old C-extension deps (`eventmachine`, `posix-spawn`, `yajl-ruby`) that won't compile against macOS Tahoe's toolchain on Ruby 3.3 or 4.0 without manual SDKROOT/openssl flag wrangling. Don't yak-shave on it, the user has accepted skipping local preview and relying on GitHub Pages' own build for visual verification. If they ask for local preview in a future session, treat it as a proper setup task and surface the cost-benefit before grinding through compile errors.

## Commit style

Look at recent commits before writing one, current style is short, present-tense, action-first:
- "Add scroll-reveal fade-up for post-list items and h2 sections"
- "Link projects from their posts and add an inline project button to post lists"

Not "feat:" prefixed, not Conventional Commits. Keep it to one line where possible.

## Things to be careful with

- **Don't reintroduce Font Awesome / Bootstrap**, the design specifically dropped them. CSS has explicit rules zeroing out leftover `.fa-stack` and `.social-media-list i`.
- **Don't widen the content column** without asking, `$content-width: 640px` is a deliberate reading-width choice.
- **Respect `prefers-reduced-motion`**, wrap any new motion in `@media (prefers-reduced-motion: no-preference)` like the existing code does.
- **Keep `_sass/` edits to a minimum**, they're easy to write and easy to have no visible effect because `main.scss` overrides them.
- **`_config.yml` `baseurl`** is empty (site hosted at root). Don't add a baseurl unless the hosting changes.
