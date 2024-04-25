# `eurocropsml-cli`

Logging setup for CLI.

**Usage**:

```console
$ eurocropsml-cli [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `acquisition`
* `datasets`

## `eurocropsml-cli acquisition`

**Usage**:

```console
$ eurocropsml-cli acquisition [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `eurocrops`

### `eurocropsml-cli acquisition eurocrops`

**Usage**:

```console
$ eurocropsml-cli acquisition eurocrops [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `config`: Print currently used config.
* `get-data`

#### `eurocropsml-cli acquisition eurocrops config`

Print currently used config.

**Usage**:

```console
$ eurocropsml-cli acquisition eurocrops config [OPTIONS] [OVERRIDES]...
```

**Arguments**:

* `[OVERRIDES]...`: Overrides to config

**Options**:

* `--config-path TEXT`: Path to config.yaml file.
* `--help`: Show this message and exit.

#### `eurocropsml-cli acquisition eurocrops get-data`

**Usage**:

```console
$ eurocropsml-cli acquisition eurocrops get-data [OPTIONS] [OVERRIDES]...
```

**Arguments**:

* `[OVERRIDES]...`: Overrides to config

**Options**:

* `--config-path TEXT`: Path to config.yaml file.
* `--help`: Show this message and exit.

## `eurocropsml-cli datasets`

**Usage**:

```console
$ eurocropsml-cli datasets [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `eurocrops`

### `eurocropsml-cli datasets eurocrops`

**Usage**:

```console
$ eurocropsml-cli datasets eurocrops [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `build-splits`
* `config`: Print currently used config.
* `download`
* `preprocess`

#### `eurocropsml-cli datasets eurocrops build-splits`

**Usage**:

```console
$ eurocropsml-cli datasets eurocrops build-splits [OPTIONS] [OVERRIDES]...
```

**Arguments**:

* `[OVERRIDES]...`: Overrides to split config

**Options**:

* `--config-path TEXT`: Path to config.yaml file.
* `--help`: Show this message and exit.

#### `eurocropsml-cli datasets eurocrops config`

Print currently used config.

**Usage**:

```console
$ eurocropsml-cli datasets eurocrops config [OPTIONS] [OVERRIDES]...
```

**Arguments**:

* `[OVERRIDES]...`: Overrides to preprocess config

**Options**:

* `--config-path TEXT`: Path to config.yaml file.
* `--help`: Show this message and exit.

#### `eurocropsml-cli datasets eurocrops download`

**Usage**:

```console
$ eurocropsml-cli datasets eurocrops download [OPTIONS] [OVERRIDES]...
```

**Arguments**:

* `[OVERRIDES]...`: Overrides to preprocess config

**Options**:

* `--config-path TEXT`: Path to config.yaml file.
* `--help`: Show this message and exit.

#### `eurocropsml-cli datasets eurocrops preprocess`

**Usage**:

```console
$ eurocropsml-cli datasets eurocrops preprocess [OPTIONS] [OVERRIDES]...
```

**Arguments**:

* `[OVERRIDES]...`: Overrides to preprocess config

**Options**:

* `--config-path TEXT`: Path to config.yaml file.
* `--help`: Show this message and exit.
