.PHONY: serve build check clean

serve:
	zola serve --base-url http://127.0.0.1

build:
	zola build

check:
	zola check

clean:
	rm -rf public
