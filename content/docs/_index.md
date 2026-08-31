+++
title = "Docs"
description = "Practical notes for moving from a fresh machine to programs running on Tenstorrent hardware."
sort_by = "weight"
template = "docs/section.html"
page_template = "docs/page.html"
+++

## Start with the path, not the API

The stack is easiest to understand from the hardware upward: bring up the card, run one known-good program, then follow compilation and execution through the layers. These notes keep that path short and make the boundaries explicit.

The documentation is written as ordinary Markdown. Add a file here, give it a `weight`, and Zola places it in the navigation automatically.
