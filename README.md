# MTA Markov

A Python package for Multi-Touch Attribution (MTA) using Markov Chains.

## Features

- Efficient implementation using Sherman-Morrison formula
- Handles large datasets with batch processing
- Supports value-based attribution
- Easy to integrate with existing marketing analytics pipelines

## Installation

```bash
pip install mta_markov
```

## Usage

```python
import pandas as pd
from mta_markov import markov_chain_attribution

# Sample data
data = {
    'path': ['A > B > C', 'A > D', 'B > C'],
    'conversions': [1, 0, 1],
    'value': [100, 0, 50],
    'null': [0, 1, 0]
}
df = pd.DataFrame(data)

# Run attribution
result = markov_chain_attribution(
    df,
    var_path='path',
    var_conv='conversions',
    var_value='value',
    var_null='null',
    use_value=True
)

print(result)
```

## Documentation

For detailed API documentation and examples, see [docs](docs/).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
