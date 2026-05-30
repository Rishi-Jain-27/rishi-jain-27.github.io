---
layout: page
title: "Posts"
permalink: /posts/
main_nav: true
---

{% for category in site.categories %}
  {% capture cat %}{{ category | first }}{% endcapture %}
  <h2 id="{{cat}}">{{ cat | capitalize }}</h2>
  {% for desc in site.descriptions %}
    {% if desc.cat == cat %}
      <p class="desc"><em>{{ desc.desc }}</em></p>
    {% endif %}
  {% endfor %}
  <ul class="posts-list">
  {% for post in site.categories[cat] %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      {% if post.project_url %}<a class="project-button" href="{{ post.project_url }}" target="_blank" rel="noopener noreferrer" title="Open project: {{ post.project_url }}">{{ post.project_label | default: "Open" }} <span class="project-button-arrow" aria-hidden="true">↗</span></a>{% endif %}
      {% if post.code_url %}<a class="project-button" href="{{ post.code_url }}" target="_blank" rel="noopener noreferrer" title="View code: {{ post.code_url }}">{{ post.code_label | default: "Code" }} <span class="project-button-arrow" aria-hidden="true">↗</span></a>{% endif %}
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
  {% if forloop.last == false %}<hr>{% endif %}
{% endfor %}
<br>
