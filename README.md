# NLP Solution accelarator
This is a solution Accelarator which uses hugging face to train on any downstream task currently it support the following downstream task

1. Multi-Classification task
2. Named Entity Recoginition


## Installing project requirements

```bash
pip install -r requirements.txt
```

## Install project package in a developer mode

```bash
pip install -e .
```

## Testing

For local unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

For an integration test on interactive cluster, use the following command:
```
dbx execute --cluster-name=<name of interactive cluster> --job=nlp_ssa-sample-integration-test
```

For a test on an automated job cluster, deploy the job files and then launch:
```
dbx deploy --jobs=nlp_ssa-sample-integration-test --files-only
dbx launch --job=nlp_ssa-sample-integration-test --as-run-submit --trace
```

## Interactive execution and development

1. `dbx` expects that cluster for interactive execution supports `%pip` and `%conda` magic [commands](https://docs.databricks.com/libraries/notebooks-python-libraries.html).
2. Please configure your job in `conf/deployment.yml` file.
2. To execute the code interactively, provide either `--cluster-id` or `--cluster-name`.
```bash
dbx execute \
    --cluster-name="<some-cluster-name>" \
    --job=job-name
```

Multiple users also can use the same cluster for development. Libraries will be isolated per each execution context.

## Preparing deployment file

Next step would be to configure your deployment objects. To make this process easy and flexible, we're using YAML for configuration.

By default, deployment configuration is stored in `conf/deployment.yml`.

## Deployment for Run Submit API

To deploy only the files and not to override the job definitions, do the following:

```bash
dbx deploy --files-only
```

To launch the file-based deployment:
```
dbx launch --as-run-submit --trace
```

This type of deployment is handy for working in different branches, not to affect the main job definition.

## Deployment for Run Now API

To deploy files and update the job definitions:

```bash
dbx deploy
```

To launch the file-based deployment:
```
dbx launch --job=<job-name>
```

This type of deployment shall be mainly used from the CI pipeline in automated way during new release.


## CICD pipeline settings

Please set the following secrets or environment variables for your CI provider:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

## Testing and releasing via CI pipeline

- To trigger the CI pipeline, simply push your code to the repository. If CI provider is correctly set, it shall trigger the general testing pipeline
- To trigger the release pipeline, get the current version from the `nlp_sa/__init__.py` file and tag the current code version:
```
git tag -a v<your-project-version> -m "Release tag for version <your-project-version>"
git push origin --tags
```

## Working with notebooks and Repos
To start working with your notebooks from Repos, do the following steps:

Add your git provider token to your user settings
Add your repository to Repos. This could be done via UI, or via CLI command below:
```
databricks repos create --url <your repo URL> --provider <your-provider>
```
This command will create your personal repository under /Repos/<username>/telco_churn. 3. To set up the CI/CD pipeline with the notebook, create a separate Staging repo:
```
dbx sync repo -d transformer-nlp-solution-accelarator_dbx --profile e2-field --use-gitignore -ep mlruns/  --no-watch
```

You can sync your local repository with databricks repos using dbx sync command
'''
dbx sync repo -d transformer-nlp-solution-accelarator_dbx --profile e2-field --use-gitignore -ep mlruns/  --no-watch
'''