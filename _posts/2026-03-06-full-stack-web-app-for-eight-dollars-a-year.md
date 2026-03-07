---
layout: post
title: "Building a Full-Stack Web App With AI and PostgreSQL — For $8/Year"
author: "Rishi Jain"
date: 2026-03-06
categories: projects
---

I recently built a full-stack, web-based application backed by PostgreSQL with AI-powered features — and the total hosting cost comes out to just **$8 a year**. Check it out at [hwagent.org](https://hwagent.org).

## The Stack

At its core, the app is a full-stack web application with a PostgreSQL database powering the backend. PostgreSQL was the natural choice — it's rock-solid, open source, and scales well beyond what I need for this project. On top of that relational foundation, I integrated AI-powered features to make the app smarter and more useful.

## AI-Powered Features

AI isn't just a buzzword here — it's woven into the functionality of the app. By leveraging AI capabilities, the application can do more with less code and deliver a better experience to users. The integration sits cleanly alongside the traditional CRUD operations backed by PostgreSQL.

## Hardening: Input Validation and XSS Fixes

Building features is one thing; shipping them securely is another. I went through the application and hardened it with:

- **Input validation** — Every user-facing input is validated on both the client and server side. Malformed or unexpected data is rejected before it ever touches the database.
- **XSS (Cross-Site Scripting) fixes** — I audited the frontend to ensure that user-generated content is properly sanitized and escaped. No raw HTML or unescaped data makes it into the DOM.

These aren't glamorous changes, but they're the kind of work that separates a toy project from something you can actually put in front of real users.

## All for $8/Year

The best part? The entire thing runs for just **$8 a year**. By being thoughtful about architecture and choosing the right providers, it's possible to run a full-stack app with a real database and AI features for less than the cost of a single lunch. Cloud doesn't have to mean expensive.

## Takeaway

You don't need a big budget to build something real. A PostgreSQL-backed, AI-enhanced, security-hardened web app is well within reach — even on a shoestring budget. Start building, ship it, and harden it. In that order.
