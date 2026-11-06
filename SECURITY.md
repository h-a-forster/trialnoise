# Security policy

Security fixes are made for the latest release.

## Reporting

Report vulnerabilities privately through the repository's **Security** tab ("Report a
vulnerability"). Do not open a public issue. Include the trialnoise version, the operating system,
the impact, and steps that reproduce it.

## Scope

trialnoise downloads two pinned datasets from huggingface.co (`trialnoise fetch`), reads local trial
tables, and writes CSV, JSON, Parquet and Markdown under the paths you pass. It calls no models.

In scope:

- reading, writing or deleting files outside the paths given, for example through a submission or
  job name in a downloaded file;
- code execution triggered by the contents of a data file;
- local paths or machine details written to committed outputs.

Out of scope: vulnerabilities in dependencies (report those upstream), and statistical disagreements
(open an ordinary issue).
