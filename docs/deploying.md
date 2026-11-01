# Running a real server

`unsaid serve` is one process with no dependencies. It holds the index in memory, and it stores nothing
about requests beyond two counters, which `GET /v1/stats` shows to anyone who asks.

## Sizing

`unsaid info` and the output of `unsaid build` give the exact figures for your index. Memory is about
the size of the two matrix files. The [benchmarks](benchmarks.md) show how time grows with the corpus.
Every search reads the whole index, so throughput falls as the index grows, and more cores help
almost linearly.

## HTTPS

Put HTTPS in front of it, for example with [Caddy](https://caddyserver.com):

```text
search.example.com {
    reverse_proxy 127.0.0.1:8377
}
```

The encryption protects queries without HTTPS, but HTTPS still matters. Without it, anyone on the
network can see that you are using this server, and they can tamper with the web interface, which
would defeat everything for people using the page.

Check that your proxy does not log anything you do not want. A request log will show the IP addresses
and times of searches, but not their contents.

## Docker

```sh
docker build -t unsaid .
docker run -p 8377:8377 -v "$PWD/my-index:/index:ro" unsaid
```

## Publish the index identifier

`unsaid info` prints an index identifier that covers the whole index. Publish it somewhere other than
the server, such as a README, and clients can check it:

```sh
unsaid search "your question" --server https://search.example.com --pin 474093d2d0d4bcb9024fb3dceb1789d1
```

That stops a server from quietly giving one person a different index.

## Rate limits

Each search makes the server do real work for anyone who asks, so a public server needs limits. Set
them in your reverse proxy. Limit by IP address, or issue API keys there, because the server itself
has no accounts. It caps queries per request and how many requests it works on at once, and nothing
more than that.

## Updating the index

Rebuild it and restart. An index has a new identifier and new hints after any change, so clients
download the changed files again. Files that did not change, such as the model, come from their cache.
