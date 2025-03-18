# Contributing Guide

Welcome to MTA Markov! We appreciate your interest in contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up development environment:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Workflow

1. Create a new branch for your feature/bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for your changes
4. Run tests:
   ```bash
   pytest
   ```
5. Commit your changes with a descriptive message
6. Push your branch to your fork
7. Create a pull request

## Code Style

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Write docstrings for all public functions
- Keep lines under 100 characters

## Testing

- Write unit tests for all new functionality
- Maintain at least 80% test coverage
- Use descriptive test names
- Test edge cases and error conditions

## Documentation

- Update API documentation for any new/changed functionality
- Add usage examples to README.md if applicable
- Keep docstrings up to date

## Reporting Issues

When reporting issues, please include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Version information
