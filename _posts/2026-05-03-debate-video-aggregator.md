---
layout: post
title: "Aggregating Every Debate Round on YouTube — and Letting AI Recommend the Best Ones"
author: "Rishi Jain"
date: 2026-05-03
categories: projects
---

Debate is one of those activities where the learning curve is brutal if you don't have access to good rounds to watch. Coaches will tell you to "watch more rounds," but the rounds are scattered across hundreds of YouTube channels — camp uploads, tournament archives, individual debater playlists — with no central place to find them and no real way to know which ones are actually worth your two hours. So I built one.

## What It Does

The app is a full-stack web application that aggregates debate videos across YouTube. Instead of asking users to hunt through channels, it pulls rounds in via the **YouTube Data API** and serves them back through embedded players — so the entire viewing experience stays inside the app while YouTube handles delivery. From a student's perspective, it looks like one giant, searchable library of every round worth watching.

On top of that aggregation layer, I built an **AI recommendation system** that surfaces the best rounds for a given user. The model takes into account the kind of debate the user is preparing for — the topic, side, style, level — and suggests rounds that are most likely to actually move the needle for them. A novice prepping for their first tournament gets very different recommendations than a varsity debater grinding for nationals, and the goal is to make digital debate accessible to students who don't have a coach feeding them rounds by hand.

## Datasets I Had to Manage

A surprising amount of the work wasn't the AI or the embeds — it was data modeling. The app maintains several datasets that all have to stay in sync:

- **Users** — accounts, preferences, watch history, and the signals that feed the recommender.
- **Admins** — a separate role with permissions to curate the catalog, flag bad uploads, and override recommendations when something obviously wrong slips through.
- **Videos and metadata** — the YouTube video ID, title, channel, topic, side, year, tournament, judges where known, and a bunch of derived fields I compute on ingest. Keeping this metadata clean is what makes the recommendations work; garbage in, garbage out.

Getting these datasets to play well together — without leaking admin-only fields to regular users, without breaking referential integrity when a YouTube video gets deleted upstream, without letting the recommender train on stale data — was the part of the project I underestimated the most.

## Why It Matters

Debate has a real access problem. Kids at well-funded programs get access to coaches, camp recordings, and curated round libraries; everyone else gets a YouTube search bar. Aggregating the rounds is the easy half — the AI recommendations are what actually expand digital debate accessibility, because they replace the part of the experience that used to require an experienced coach sitting next to you saying *"watch this one next."*

## Takeaway

The interesting engineering wasn't any one piece — it was stitching together a third-party content API, a recommendation model, and a multi-role data model into something a 14-year-old novice and a coach can both open up and immediately get value out of. That's the bar I care about: does it actually help someone get better at debate? Everything else is implementation detail.
