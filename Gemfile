source 'https://rubygems.org'

# Local development uses plain Jekyll 4 (fast, prebuilt binary gems, no native
# compile). GitHub Pages still builds the deployed site with its own
# `github-pages` environment and ignores this Gemfile, so production is
# unaffected. The plugins here mirror the `plugins:` list in _config.yml.
gem 'jekyll', '~> 4.3'

group :jekyll_plugins do
  gem 'jekyll-sitemap'
  gem 'jekyll-paginate'
end

# Ruby 3+ dropped webrick from the standard library; Jekyll's local server needs
# it explicitly for `jekyll serve`.
gem 'webrick', '~> 1.8'
