# relativity.cc

A deliberately small Zola site: HTML, CSS, inline SVG, and no client-side JavaScript.

## Why Zola?

- single binary
- no `node_modules`
- Markdown + templates when the site grows
- outputs plain static files
- first-class GitHub Pages deployment path
- fast enough that the build is basically invisible

## Local development

Install Zola 0.23.4 or newer, then:

```sh
make serve
```

The site will be available at the local URL printed by Zola.

Useful commands:

```sh
make check
make build
```

## Deploy to GitHub Pages

The repository already contains `.github/workflows/pages.yml`.

1. Push the repository to GitHub with `main` as the default branch.
2. Open **Settings → Pages**.
3. Set **Build and deployment → Source** to **GitHub Actions**.
4. Push a commit. The workflow builds with Zola and publishes the `public/` artifact.

The workflow asks GitHub Pages for the actual base URL, so it works both as a project site (`user.github.io/repo`) and later with a custom domain.

## Custom domain

When you are ready to use `relativity.cc`, add the custom domain in **Settings → Pages** and configure the DNS records GitHub shows you. The build will pick up the custom base URL automatically.

## Project structure

- `templates/`: page and shared-layout templates.
- `content/`: Zola sections and page metadata.
- `static/style.css`: palette and layout.
- `static/favicon.svg`: the light-cone `R` mark.
