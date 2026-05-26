---
layout: post
title: "A full-stack web app for eight dollars a year"
author: "Rishi Jain"
date: 2026-03-06
categories: projects
project_url: https://hwagent.org
project_label: Visit
---

I built a web app with a PostgreSQL backend and some AI features baked in, and the entire thing runs for about $8 a year. It's live at [hwagent.org](https://hwagent.org).

## The stack

Postgres on the backend, a normal web frontend, AI calls wired in where they earn their keep. Postgres because I already know it and it does what I need; the AI bits sit next to ordinary CRUD rather than replacing it.

## What the AI actually does

It's not the whole app, it's a few specific features where having a model in the loop is meaningfully better than not having one. The rest is database queries and forms.

## Hardening

After the features worked I went back and tightened the obvious things: validation on both sides of every form, escaping every place user content gets rendered, no raw HTML reaching the DOM. None of this is exciting. It's also the part of the project that lets me leave it up without checking on it.

## Where the $8 goes

A small managed Postgres instance plus a domain. That's it. Nothing else is paid. The architecture is boring enough to fit on a free or near-free tier of whatever provider you trust.

## Takeaway

A real database, a real frontend, a few AI features, and a domain you can hand someone, for the price of a single lunch per year. The lesson isn't that hosting is cheap (it is). It's that most of the cost in side projects is paying for things you don't actually use.
