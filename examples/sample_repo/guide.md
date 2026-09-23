# Repository guide

This fixture describes the sample repository.

## Parsing

Python files keep their class and function structure.

```python
# A heading inside a code fence is code, not document structure.
def example():
    return True
```

### Safety

Malformed source is handled without discarding valid neighboring definitions.
