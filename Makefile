.PHONY: serve build check clean

serve:
	zola serve

build:
	zola build

check:
	zola check

clean:
	rm -rf public
