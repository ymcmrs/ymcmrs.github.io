# Personal Site

## Installation

On Linux

1. Install all necesary libraries with sudo apt-get install ruby-full build-essential zlib1g-dev`
2. `echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc`
3. `echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc`
4. `echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc`
5. `source ~/.bashrc`
6. `sudo gem install bundler --version '1.16'` 
7. `bundle install` (if this doesn't work try installing before this running `sudo gem install bundler`)

and run it with `bundle exec jekyll serve`



on Mac:

1. Install jekyll with `gem install bundler jekyll`. This will take a while the first time.
2. Run `bundle exec jekyll serve`.

## Phantom for Jekyll

A minimalist, responsive portfolio theme for [Jekyll](http://jekyllrb.com/) with Bootstrap.

![preview](preview.jpg)

[See it in action](http://jamigibbs.github.io/phantom/).

## Fancy using it for your own site?

Here are some steps to get you started:

1. Clone this repo and cd into the directory:

  ```bash
  git clone https://github.com/jamigibbs/phantom.git your-dir-name && cd your-dir-name
  ```

2. Run Jekyll:

  ```bash
  bundle exec jekyll serve
  ```

  _Don't have Jekyll yet? [Get `er installed then!](http://jekyllrb.com/docs/installation/)_

3. Visit in your browser at:

  `http://127.0.0.1:4000`

## Launching with Github Pages :rocket:

Jekyll + Github pages is a marriage made in heaven. You can [use your own custom domain name](https://help.github.com/articles/setting-up-a-custom-domain-with-github-pages/) or use the default Github url (ie. http://username.github.io/repository) and not bother messing around with DNS settings.

## Theme Features

### Navigation

Navigation can be customized in `_config.yml` under the `nav_item` key. Default settings:

```yaml
nav_item:
    - { url: '/', text: 'Home' }
    - { url: '/about', text: 'About' }
```

Set the `nav_enable` variable to false in `_config.yml` to disable navigation.

### Contact Form

You can display a contact form within the modal window template. This template is already setup to use the [Formspree](https://formspree.io) email system. You'll just want to add your email address to the form in `/_includes/contact-modal.html`.

Place the modal window template in any place you'd like the user to click for the contact form.
The template will display a link to click for the contact form modal window:

```liquid
{% include contact.html %}
{% include contact-modal.html %}
```

### Animation Effects

Animations with CSS classes are baked into the theme. To animate a section or element, simply add the animation classes:

```html
<div id="about-me" class="wow fadeIn">
  I'm the coolest!
</div>
```

For a complete list of animations, see the [animation list](http://daneden.github.io/animate.css/).

### Pagination

By default, pagination on the home page will activate after 10 posts. You can change this within `_config.yml`. You can add the pagination to other layouts with:

```liquid
  {% for post in paginator.posts %}
    {% include post-content.html %}
  {% endfor %}

  {% include pagination.html %}
```

Read more about the [pagination plugin](http://jekyllrb.com/docs/pagination/).

## Credit

* Bootstrap, http://getbootstrap.com/, (C) 2011 - 2016 Twitter, Inc., [MIT](https://github.com/twbs/bootstrap/blob/master/LICENSE)

* Wow, https://github.com/matthieua/WOW, (C) 2014 - 2016 Matthieu Aussaguel
, [GPL](https://github.com/matthieua/WOW#open-source-license)

* Animate.css, https://github.com/daneden/animate.css, (C) 2016 Daniel Eden, [MIT](https://github.com/daneden/animate.css/blob/master/LICENSE)
