# Changelog

## [0.1.1] - 2026-05-29

### Added
- CLAUDE.md with project instructions, architecture docs, commands, and skill routing
- Traefik static config for production reverse proxy (Host rule, TLS, Let's Encrypt resolver)
- Python design patterns skill (KISS, SRP, composition over inheritance, dependency injection)
- Skills lock file for reproducible skill setup

### Changed
- Docker Compose: commented out direct port mapping in favor of Traefik proxy