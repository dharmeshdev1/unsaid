# Reporting a security problem

Please report vulnerabilities privately through
[GitHub's security advisories](https://github.com/dharmeshdev1/unsaid/security/advisories/new), not in
a public issue. You can expect a first reply within a week.

Things that would be especially valuable to hear about:

- Any way for a server to learn something about a query, including from timing, sizes or client behaviour
- A problem with the parameters or the estimate in [`docs/security.md`](docs/security.md)
- A way to crash or exhaust a client with files from a malicious server
- A difference between the protocol as documented and as implemented

Limitations already listed in [`docs/security.md`](docs/security.md) are known, but better measurements
of them are very welcome as ordinary issues.
