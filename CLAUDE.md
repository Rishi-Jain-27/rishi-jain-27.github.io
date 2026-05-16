# CLAUDE.md

Notes for Claude when working on this site. Living doc — keep it up to date as the site evolves.

## What this site is

Rishi Jain's personal site + blog. Jekyll, originally built on the Centrarium theme but heavily customized. Hosted at https://rishi-jain-27.github.io via GitHub Pages.

Optimized for three audiences, in roughly this order:
1. **Readers** of long-form writing / blog posts.
2. **People evaluating projects** — each post can link out to a live project via the inline pill button.
3. **Professional landings** — recruiters and collaborators reading the resume / about page.

Not a casual scratchpad. Posts get edited and polished.

## Style direction

Current direction: **dark + warm-gold, slim, code-flavored.** Keep this base. Refinements welcome — propose small tweaks (palette nudges, type, spacing, motion) when they fit, but don't redesign without asking.

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
$accent    #d4af7a   warm gold — THE color
$accent-hi #ebc999   hover/lifted accent
```

Background uses two soft radial gradients of `$accent` over `$bg` for ambient glow; `background-attachment: fixed`.

### Typography

- **Sans:** system stack with Inter preferred (`-apple-system, BlinkMacSystemFont, "Inter", ...`). Body weight `300`.
- **Mono:** `ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, ...`. Used for **accents** — section markers, project pills, tech chips, code.
- Headings: weight 500–600, letter-spacing slightly negative (`-0.01em` / `-0.02em`).
- Note: `_sass/base/_variables.scss` defines an *older* light-theme (Open Sans / Roboto Slab) — that's legacy from the Centrarium theme and is overridden by [css/main.scss](css/main.scss). Don't edit the old vars expecting visible change.

### Recurring visual signatures (preserve these)

- **`// 01` section markers** — auto-numbered via CSS counters on `.page .post-content h2` and `.post .post-content h2`. The `//` makes them feel like code comments.
- **Accent left-bar slide-in** on post-list hover (`.post-list > li::before`). Hover also nudges the item right by 0.7rem.
- **Inline project pill** next to post titles when frontmatter has `project_url` — small uppercase mono badge with `↗`. See "Posts" below.
- **Slim accent diamond divider** (`◆`) instead of the original Centrarium horizontal rule.
- **Tech-tile grid** on the home hero — monochrome simpleicons CDN icons painted via CSS mask + `--icon-color`. Non-iconed tiles use a text glyph (e.g. `{ }`, `▦`, `✦`).
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
  head.html             <head>, meta, favicons, OG, AND the scroll-reveal IntersectionObserver script
  header.html           top nav with .logo + .navigation-menu (populated from nav_links.html)
  nav_links.html        iterates site.pages where main_nav: true
  footer.html           three-column footer: nav / contact / signature
  page_divider.html     the ◆ divider
_posts/                 YYYY-MM-DD-slug.md — blog content
_sass/                  Centrarium legacy SCSS (mostly overridden — see note)
css/main.scss           ★ the real stylesheet. ~690 lines, defines theme tokens.
js/                     empty — JS lives inline in _includes/head.html
index.html              home: hero (photo + title + tech grid) + paginated post list
posts.md                /posts/ — posts grouped by category
resume.md               /resume/ — long markdown resume
about.md                /about/ — short bio
assets/                 profile-hero.png, header_image.jpg, logo.png, /icons/ favicons
feed.xml                RSS
Gemfile / Gemfile.lock  github-pages gem
```

### About `_sass/`

`_sass/base/_variables.scss`, `_layout.scss`, `bourbon/`, `neat/` are all **inherited from the Centrarium theme**. They define a light-mode palette and the old layout. `css/main.scss` mostly overrides all of it (and even hides leftover Font Awesome icons it can't suppress). **Default to editing `css/main.scss`**, not `_sass/`. If you find yourself wanting to fix something in `_sass/`, first check whether `main.scss` already overrides it — usually it does.

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
project_url: https://example.com    # optional — adds inline pill
project_label: "Visit"               # optional, defaults to "Open"
math: true                           # optional — loads MathJax for LaTeX (see "Math" below)
---
```

Body is kramdown. Use `## Section` for top-level sections (will auto-get `// 01`, `// 02` markers). Use `**bold**` sparingly — bold gets `$fg` (brighter) so it pops in the muted body text.

#### Writing voice (read the existing posts before drafting a new one)

The published posts have a consistent voice. Match it.

- **Title:** strong noun-phrase + em-dash subtitle that names a thesis or paradox. Often inverts a cliché. ("MNIST Was the Excuse — Learning to Run Real Experiments in PyTorch", "FER2013 — When Transfer Learning Loses to a Smaller Model".) Avoid generic "Building X" / "How I Built X" titles.
- **Open with the thesis or the surprising result**, not setup or background. The first paragraph also tends to drop the GitHub link and any cross-references to related posts.
- **Section headings are noun phrases that make a claim**, not generic labels. "Why the Bigger Model Lost" beats "Results"; "The Three Things That Go Wrong" beats "Common Errors".
- **Self-aware about clichés.** Posts often name the conventional wisdom they're pushing against ("tutorials are a trap", "always start with transfer learning", "MNIST is solved") and answer it.
- **Specific over abstract.** Name hyperparameters, file paths, gem/library versions, exact accuracy numbers. Concrete details > general principles.
- **Honest about failures.** "I did not internalize it." / "I was surprised for about an afternoon, and then it made sense." First-person, mildly self-deprecating, never preachy or didactic.
- **Math and technical depth are welcome — and expected on technical topics.** The personal-narrative voice is the texture, not the substance. If a post is about a technical subject (an algorithm, an architecture, a derivation), it should actually *do* the technical work — write the equation, derive the gradient, work the concrete example, name the algorithm with a precise enough description that a reader could reimplement it. Trials/struggles are welcome alongside, not as a substitute. When in doubt, lean technical.
- **End with a short `## Takeaway`** (or `## What's Next` for project posts with a roadmap). Two or three sentences. Not a summary — a final claim.
- **Post length:** narrative-only posts target ~600–1000 words (cut past 1200). Technical posts with derivations or math can run longer — up to ~2000 words is fine if the math earns it. Cut filler, never cut the math.
- No emojis. No "I hope this helps." No advice paragraphs aimed at the reader.

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

`main_nav: true` is the magic — [_includes/nav_links.html](_includes/nav_links.html) iterates `site.pages` and only includes pages with this flag.

### Math (LaTeX via MathJax)

The site loads MathJax v3 from a CDN, **opt-in per page** via `math: true` in frontmatter. Pages without that flag don't fetch MathJax at all.

- **Inline math:** `\( ... \)` — e.g. `\( h_t = f(W h_{t-1} + U x_t) \)`.
- **Display math:** `\[ ... \]` on its own line(s) — e.g.:
  ```
  \[
  \frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
  \]
  ```
- **Do not use `$ ... $` for inline** (collides with currency in prose) and **do not use `$$ ... $$`** (kramdown's math handling is disabled in [_config.yml](_config.yml), but `\[ \]` is the convention here regardless).
- MathJax is configured to skip `pre` and `code` blocks, so backtick code samples render literally — useful for showing the LaTeX source itself.
- Equation numbering is off (`tags: 'none'`). If a future post needs numbered equations, flip that in [_includes/head.html](_includes/head.html).
- Display equations get a horizontal scrollbar on overflow rather than blowing past the 640px content column — see the `mjx-container[display="true"]` rule in [css/main.scss](css/main.scss).

### Scroll-reveal targets

The IntersectionObserver in [_includes/head.html](_includes/head.html) observes:
- `.post-list > li`
- `.page .post-content > h2`
- `.post .post-content > h2`

If you add new content types that should fade in on scroll, either extend the selector list there or reuse one of these classes.

### Tech tiles on home

Defined inline in [index.html](index.html). Two flavors:
- **With icon:** `<a class="tech-tile">` with `<span class="tech-icon" style="--icon-url: url('https://cdn.simpleicons.org/<name>'); --icon-color: #HEX;">`. Icon comes from simpleicons.org CDN as an SVG, painted via CSS mask.
- **Without icon:** `<a class="tech-tile tech-tile--noicon">` with `<span class="tech-icon-text">` containing a glyph / text (e.g. `{ }`, `▦`, `✦`, `⚠`).

Keep the grid feeling curated — these should be technologies Rishi actually uses, not a "look how many logos I can fit" wall.

### Project linking

A blog post about a project should:
1. Add `project_url:` (and optionally `project_label:`) to frontmatter — adds the inline pill to post listings on the home page and `/posts/`.
2. Link to the project inside the post body too (early, usually in the intro paragraph).

## Source material workflow

`source-material/` is the drop zone for raw input for upcoming blog posts. The user puts things there (notes, transcripts, screenshots, READMEs, exported chat logs, links in a `.txt`, etc.); Claude reads them, drafts the post, and then the source files get deleted once the post is in `_posts/`.

- The directory is **gitignored** (except for `.gitkeep`) — material stays local and never lands in commits.
- The directory is in `_config.yml` `exclude:` — Jekyll won't try to process it.
- When the user asks to write about something in `source-material/`, list the directory, read the files, draft the post in `_posts/` following the conventions above, and then **ask before deleting** the source files. Do not delete until the user has reviewed the draft and confirmed.
- If multiple unrelated items are in `source-material/` at once, ask which one to write about rather than guessing.

## Local development

```
bundle install        # one time
bundle exec jekyll serve
# → http://localhost:4000
```

`_site/` is the build output. It's tracked but should not be hand-edited (Jekyll regenerates it).

**Heads up — local preview is currently not working on this machine** (as of 2026-05-16). The `github-pages` gem pins old C-extension deps (`eventmachine`, `posix-spawn`, `yajl-ruby`) that won't compile against macOS Tahoe's toolchain on Ruby 3.3 or 4.0 without manual SDKROOT/openssl flag wrangling. Don't yak-shave on it — the user has accepted skipping local preview and relying on GitHub Pages' own build for visual verification. If they ask for local preview in a future session, treat it as a proper setup task and surface the cost-benefit before grinding through compile errors.

## Commit style

Look at recent commits before writing one — current style is short, present-tense, action-first:
- "Add scroll-reveal fade-up for post-list items and h2 sections"
- "Link projects from their posts and add an inline project button to post lists"

Not "feat:" prefixed, not Conventional Commits. Keep it to one line where possible.

## Things to be careful with

- **Don't reintroduce Font Awesome / Bootstrap** — the design specifically dropped them. CSS has explicit rules zeroing out leftover `.fa-stack` and `.social-media-list i`.
- **Don't widen the content column** without asking — `$content-width: 640px` is a deliberate reading-width choice.
- **Respect `prefers-reduced-motion`** — wrap any new motion in `@media (prefers-reduced-motion: no-preference)` like the existing code does.
- **Keep `_sass/` edits to a minimum** — they're easy to write and easy to have no visible effect because `main.scss` overrides them.
- **`_config.yml` `baseurl`** is empty (site hosted at root). Don't add a baseurl unless the hosting changes.
