---
layout: post
title: "A YouTube debate-video aggregator with AI recommendations"
author: "Rishi Jain"
date: 2026-05-03
categories: projects
project_url: https://debate-search.com
project_label: Visit
code_url: https://github.com/Rishi-Jain-27/debate-search
code_label: Code
---

Every coach tells debaters the same thing, which is to watch more rounds, and the trouble with that advice is that the rounds are scattered across hundreds of YouTube channels, the camp uploads and tournament archives and individual playlists, with no central index and no real way to know in advance which two-hour video is worth your evening. So I built the index, at [debate-search.com](https://debate-search.com), with the code on [GitHub](https://github.com/Rishi-Jain-27/debate-search).

## What it does

The app pulls debate videos from across YouTube through the Data API and serves them through embedded players, so the viewing stays inside the app while YouTube quietly handles the streaming, and from a student's side it reads as one large searchable library rather than a scavenger hunt across channels. Sitting on top of that catalogue is a recommendation system that takes the kind of round a user is preparing for, the topic and side and style and level, and surfaces the rounds most likely to help, so a novice walking into their first tournament gets very different suggestions from a varsity debater grinding for nationals. The aim is to give a student without a coach something close to the thing a coach provides, which is someone pointing at the next round they ought to watch.

## The datasets I had to keep in sync

Most of the work turned out to be neither the recommender nor the embeds but the data modelling, since three things all had to stay consistent with one another. There are the users, which means accounts and preferences and watch history, the signals the recommender learns from. There are the admins, a separate role with permission to curate the catalogue, flag bad uploads, and override a recommendation when something obviously wrong slips through. And there are the videos and their metadata, the YouTube video ID and title and channel and topic and side and year and tournament, the judges where they are known, and the derived fields I compute on ingest, which is the clean input the recommender leans on entirely.

Getting those three to live together was the part I most underestimated. The admin-only fields cannot leak to a regular user, referential integrity cannot break when a YouTube video gets deleted upstream out from under you, and the recommender cannot be allowed to train on stale data, and holding all three of those true at once was genuinely the hard half of the project.

## Why it matters

Debate has an access problem that is easy to name and harder to fix. The students at well-funded programmes get coaches and camp recordings and curated round libraries, and everyone else gets a YouTube search bar. Aggregating the rounds is the straightforward half of closing that gap, and the recommendations are the half that stands in for the experienced coach sitting next to you and saying watch this one next.

## Takeaway

The engineering worth talking about was the stitching: a third-party content API, a recommendation model, and a multi-role data model joined into one thing that a fourteen-year-old novice and a coach can both open and immediately get something out of, which was the whole point of building it.
