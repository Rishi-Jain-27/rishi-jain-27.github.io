---
layout: post
title: "A YouTube debate-video aggregator with AI recommendations"
author: "Rishi Jain"
date: 2026-05-03
categories: projects
project_url: https://debate-search.com
project_label: Visit
---

Coaches tell debaters to watch more rounds. The problem is that the rounds live across hundreds of YouTube channels (camp uploads, tournament archives, individual playlists) with no central index and no real way to know in advance which ones are worth two hours of your evening. So I built one, at [debate-search.com](https://debate-search.com).

## What it does

The app pulls debate videos from across YouTube via the Data API and serves them through embedded players, so the viewing experience stays inside the app while YouTube handles the actual streaming. From a student's perspective it looks like one large, searchable library.

Sitting on top of the catalog is a recommendation system that takes the kind of debate a user is preparing for (topic, side, style, level) and surfaces the rounds most likely to help. A novice prepping for their first tournament gets very different suggestions than a varsity debater grinding for nationals. The goal is to give students without a coach something close to what a coach provides: someone pointing at the next round they should watch.

## Datasets I had to manage

Most of the work wasn't the recommender or the embeds. It was data modeling. Three things had to stay in sync:

- **Users.** Accounts, preferences, watch history, the signals the recommender trains on.
- **Admins.** A separate role with permission to curate the catalog, flag bad uploads, and override recommendations when something obviously wrong slips through.
- **Videos and metadata.** YouTube video ID, title, channel, topic, side, year, tournament, judges where known, plus derived fields I compute on ingest. Clean metadata is the input the recommender depends on.

Getting these to play well together without leaking admin-only fields to regular users, without breaking referential integrity when a YouTube video gets deleted upstream, and without letting the recommender train on stale data was the part I most underestimated.

## Why it matters

Debate has an access problem. Kids at well-funded programs get coaches, camp recordings, and curated round libraries. Everyone else gets a YouTube search bar. Aggregating the rounds is the easy half. The recommendations are the half that replaces an experienced coach sitting next to you saying "watch this one next."

## Takeaway

The interesting engineering was stitching three pieces together: a third-party content API, a recommendation model, and a multi-role data model, into something a 14-year-old novice and a coach can both open and immediately get something out of.
