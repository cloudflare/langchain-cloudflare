# Changelog

All notable changes to the Langchain Cloudflare packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## langchain-cloudflare

### [0.1.9]

#### Changed
- Updated Python version requirement from `>=3.9,<4.0` to `>=3.10,<4.0`
- Updated `langchain-core` dependency from `^0.3.15` to `>=0.3.15,<2.0.0` for better compatibility
- Updated `langchain-tests` dependency from `^0.3.17` to `>=0.3.17` for better compatibility

## langgraph-checkpoint-cloudflare-d1

### [0.1.4]

#### Added
- Optional logging configuration via `enable_logging` parameter in both `CloudflareD1Saver` and `AsyncCloudflareD1Saver` classes. Defaults to `False` (opt-in).

#### Security
- CVE-2025-64439 security fixes

### [0.1.0] - Initial
- Initial release


## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

---

## Release Template

When creating a new release, copy this template and fill in the details:

## [Package Name] - [Version] - [Date]

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes
