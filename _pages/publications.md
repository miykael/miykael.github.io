---
layout: page
permalink: /publications/
title: Publications
description: Publications (patents, thesis, papers and open source projects) in reversed chronological order.
years: [2026, 2023, 2021, 2020, 2019, 2018, 2017, 2015, 2014, 2012]
patent_years: [2026, 2025, 2024]
nav: true
---

<div class="publications">

<h1>Patents</h1>

{% for y in page.patent_years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @patent[year={{y}}] %}
{% endfor %}

<h1>Papers & Other</h1>

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @article[year={{y}}] %}
  {% bibliography -f papers -q @inproceedings[year={{y}}] %}
  {% bibliography -f papers -q @misc[year={{y}}] %}
{% endfor %}

</div>
